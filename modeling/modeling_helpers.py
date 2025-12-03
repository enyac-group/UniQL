import math
import json
from tqdm import tqdm

import torch
from einops import rearrange

from safetensors import safe_open
from transformers.quantizers.quantizers_utils import get_module_from_name
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files

from quantize.quantize_helpers import w4a16_quantize
from quantize.quantize_helpers import w4a16_permute_scales
from quantize.quantize_helpers import w4a16_unpermute_scales
from quantize.quantize_helpers import get_w4a16_weight_perms
from quantize.quantize_helpers import get_w4a16_permute_weights
from quantize.quantize_helpers import unpack_w4a16_permute_weights
from quantize.quantize_helpers import MARLIN_QQQ_MIN_THREAD_N, MARLIN_QQQ_MAX_PARALLEL


"""
https://github.com/huggingface/transformers/blob/v4.53.1/src/transformers/modeling_utils.py#L729
Because transformers library only loads one weight at a time, we can not get the weight and scales to perform dequantize and quantize.
Therefore, we can not use the pre-load hook to load and slice the weight.
We slice the weight and scales when we load the state_dict
"""
def load_config_hf(model_name, cache_dir=None):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, cache_dir=cache_dir, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))


def load_state_dict_hf(model_name, device=None, cache_dir=None):
    # First try to load the safetensors index file
    resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME, cache_dir=cache_dir,
                                        _raise_exceptions_for_missing_entries=False)
    
    if resolved_archive_file is not None:
        # This is a sharded safetensors model
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name, resolved_archive_file
        )
        state_dict = {}
        for sharded_file in resolved_archive_file:
            with safe_open(sharded_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict
    
    # Fall back to single safetensors file
    resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_NAME, cache_dir=cache_dir,
                                        _raise_exceptions_for_missing_entries=False)
    # Load using safetensors
    with safe_open(resolved_archive_file, framework="pt", device="cpu") as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _pre_load_up_gate_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, load_shape):
    if "weight" in state_dict:
        assert load_shape[1] == state_dict['weight'].shape[1], "hidden_size dimension mismatch" # [Dout, Din], Din should be hidden_size
        weight = state_dict['weight']
        loaded_weight = weight[:load_shape[0], :] 
        state_dict['weight'] = loaded_weight


def _pre_load_down_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, load_shape):
    if "weight" in state_dict:
        assert load_shape[0] == state_dict['weight'].shape[0], "hidden_size dimension mismatch" # [Dout, Din], Dout should be hidden_size
        weight = state_dict['weight']
        loaded_weight = weight[:, :load_shape[1]] 
        state_dict['weight'] = loaded_weight


def _pre_load_qk_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, load_shape, head_dim, reduced_head_dim):
    if "weight" in state_dict:
        assert load_shape[1] == state_dict['weight'].shape[1], "hidden_size dimension mismatch" # [Dout, Din], Din should be hidden_size
        weight = state_dict['weight']
        weight = rearrange(weight, '(n d) h -> n d h', d=head_dim)  # [dout din] -> [n, d, h], h: hidden_size
        half = head_dim // 2
        hd_half = reduced_head_dim // 2
        loaded_weight = torch.cat([weight[:, :hd_half, :],
                                   weight[:, half:half+hd_half, :]], dim=1)
        loaded_weight = rearrange(loaded_weight, 'n d h -> (n d) h', d=reduced_head_dim)
        state_dict['weight'] = loaded_weight
    if "bias" in state_dict:
        bias = state_dict['bias']
        bias = rearrange(bias, '(n d) -> n d', d=head_dim)
        half = head_dim // 2
        hd_half = reduced_head_dim // 2
        loaded_bias = torch.cat([bias[:, :hd_half], bias[:, half:half+hd_half]], dim=1)
        loaded_bias = rearrange(loaded_bias, 'n d -> (n d)')
        state_dict['bias'] = loaded_bias


def _pre_load_v_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, load_shape, head_dim, reduced_head_dim):
    if "weight" in state_dict:
        assert load_shape[1] == state_dict['weight'].shape[1], "hidden_size dimension mismatch" # [Dout, Din], Din should be hidden_size
        weight = state_dict['weight']
        weight = rearrange(weight, '(n d) h -> n d h', d=head_dim)  # [dout din] -> [n, d, h], h: hidden_size
        loaded_weight = weight[:, :reduced_head_dim, :] 
        loaded_weight = rearrange(loaded_weight, 'n d h -> (n d) h', d=reduced_head_dim)
        state_dict['weight'] = loaded_weight
    # The v_proj bias in Qwen2 is fused with the o_proj bias


def _pre_load_o_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, load_shape, head_dim, reduced_head_dim):
    if "weight" in state_dict:
        assert load_shape[0] == state_dict['weight'].shape[0], "hidden_size dimension mismatch" # [Dout, Din], Dout should be hidden_size
        weight = state_dict['weight']
        weight = rearrange(weight, 'h (n d) -> h n d', d=head_dim)  # [dout din] -> [h, n, d], h: hidden_size
        loaded_weight = weight[:, :, :reduced_head_dim] 
        loaded_weight = rearrange(loaded_weight, 'h n d -> h (n d)', d=reduced_head_dim)
        state_dict['weight'] = loaded_weight


def get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim):
    base_indices = torch.arange(n_groups) * (num_heads // n_groups) * head_dim # [ngroups]
    base_indices = base_indices.unsqueeze(-1) + torch.arange(num_heads // n_groups).unsqueeze(0) * head_dim # [ngroups, nheads]
    topk_head_indices = torch.arange(topk_head_dim).unsqueeze(0).repeat(n_groups, num_heads // n_groups, 1) # [ngroups, nheads, topk_head_dim]
    topk_head_indices = base_indices.unsqueeze(-1) + topk_head_indices # [ngroups, nheads, topk_head_dim]
    topk_head_indices = topk_head_indices.reshape(-1) # [ngroups * nheads * topk_head_dim]
    return topk_head_indices

def get_topk_ssm_state_indices(n_groups, ssm_state_size, topk_ssm_state):
    base_indices = torch.arange(n_groups) * ssm_state_size # [ngroups]
    topk_ssm_state_indices = torch.arange(topk_ssm_state).unsqueeze(0).repeat(n_groups, 1) # [ngroups, topk_ssm_state]
    topk_ssm_state_indices = base_indices.unsqueeze(-1) + topk_ssm_state_indices # [ngroups, topk_ssm_state]
    topk_ssm_state_indices = topk_ssm_state_indices.reshape(-1) # [ngroups * topk_ssm_state]
    return topk_ssm_state_indices


def _pre_load_in_proj_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs,
                   n_groups, num_heads, intermediate_size, ssm_state_size,
                   load_intermediate_size, load_ssm_state_size):
    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    if "weight" in state_dict:
        weight = state_dict['weight']
        device = weight.device
        topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(device)
        topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(device)
        z_start = 0
        z_indices = z_start + topk_head_indices
        w_z = weight[z_indices, :] # [Dout, Din]
        x_start = z_start + intermediate_size
        x_indices = x_start + topk_head_indices
        w_x = weight[x_indices, :]
        b_start = x_start + intermediate_size
        b_indices = b_start + topk_ssm_state_indices
        w_b = weight[b_indices, :]
        c_start = b_start + n_groups*ssm_state_size
        c_indices = c_start + topk_ssm_state_indices
        w_c = weight[c_indices, :]
        dt_start = c_start + n_groups*ssm_state_size
        w_dt = weight[dt_start:, :]
        state_dict['weight'] = torch.cat([w_z, w_x, w_b, w_c, w_dt], dim=0)


def _pre_load_conv1d_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs,
                   n_groups, num_heads, intermediate_size, ssm_state_size,
                   load_intermediate_size, load_ssm_state_size):

    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    if "weight" in state_dict:
        weight = state_dict['weight']
        device = weight.device
        topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(device)
        topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(device)
        x_start = 0
        x_indices = x_start + topk_head_indices
        conv_w_x = weight[x_indices, :, :]
        b_start = x_start + intermediate_size
        b_indices = b_start + topk_ssm_state_indices
        conv_w_b = weight[b_indices, :, :]
        c_start = b_start + n_groups*ssm_state_size
        c_indices = c_start + topk_ssm_state_indices
        conv_w_c = weight[c_indices, :, :]
        conv_weight = torch.cat([conv_w_x, conv_w_b, conv_w_c], dim=0)
        state_dict['weight'] = conv_weight
    if "bias" in state_dict:
        bias = state_dict['bias']
        device = bias.device
        topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(device)
        topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(device)
        x_start = 0 
        x_indices = x_start + topk_head_indices
        conv_bias_x = bias[x_indices]
        b_start = x_start + intermediate_size
        b_indices = b_start + topk_ssm_state_indices
        conv_bias_b = bias[b_indices]
        c_start = b_start + n_groups*ssm_state_size
        c_indices = c_start + topk_ssm_state_indices
        conv_bias_c = bias[c_indices]
        conv_bias = torch.cat([conv_bias_x, conv_bias_b, conv_bias_c], dim=0)
        state_dict['bias'] = conv_bias


def _pre_load_norm_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, n_groups, num_heads, intermediate_size, load_intermediate_size):

    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    if "weight" in state_dict:
        weight = state_dict['weight']
        device = weight.device
        topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(device)
        loaded_weight = weight[topk_head_indices]
        state_dict['weight'] = loaded_weight


def _pre_load_out_proj_hook(state_dict, prefix, local_metadata,
                   strict, missing_keys, unexpected_keys, error_msgs, n_groups, num_heads, intermediate_size, load_intermediate_size):
    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    if "weight" in state_dict:
        weight = state_dict['weight'] # [Dout, Din]
        device = weight.device
        topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(device)
        loaded_weight = weight[:, topk_head_indices]
        state_dict['weight'] = loaded_weight



def slice_quantized_weights(model, model_helper, loaded_model):

    # replace the layer to simple versions for collecting the scaling factors
    layers = model_helper.get_layers(model)
    for layer_idx, layer in enumerate(tqdm(layers, desc=f"Slicing quantized weights")):
        for name, module in layer.named_children():
            class_name = module.__class__.__name__
            if class_name in [model_helper.mlp_w4a16_uniql_class_name]:
                mlp_key = model_helper.get_mlp_key(layer_idx)
                up_proj_key = f"{mlp_key}.up_proj"
                up_proj_sliced_weight, up_proj_sliced_scale, up_proj_sliced_workspace = slice_w4a16_up_gate_proj(
                    loaded_model[up_proj_key + ".weight"], loaded_model[up_proj_key + ".scale"], loaded_model[up_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.intermediate_size, module.intermediate_size, module.up_proj.weight.shape)
                loaded_model[up_proj_key + ".weight"] = up_proj_sliced_weight
                loaded_model[up_proj_key + ".scale"] = up_proj_sliced_scale
                loaded_model[up_proj_key + ".workspace"] = up_proj_sliced_workspace

                gate_proj_key = f"{mlp_key}.gate_proj"
                if gate_proj_key + ".weight" in loaded_model.keys():
                    gate_proj_sliced_weight, gate_proj_sliced_scale, gate_proj_sliced_workspace = slice_w4a16_up_gate_proj(
                        loaded_model[gate_proj_key + ".weight"], loaded_model[gate_proj_key + ".scale"], loaded_model[gate_proj_key + ".workspace"],
                        model.config.hidden_size, model.config.intermediate_size, module.intermediate_size, module.gate_proj.weight.shape)
                    loaded_model[gate_proj_key + ".weight"] = gate_proj_sliced_weight
                    loaded_model[gate_proj_key + ".scale"] = gate_proj_sliced_scale
                    loaded_model[gate_proj_key + ".workspace"] = gate_proj_sliced_workspace

                down_proj_key = f"{mlp_key}.down_proj"
                down_proj_sliced_weight, down_proj_sliced_scale, down_proj_sliced_workspace = slice_w4a16_down_proj(
                    loaded_model[down_proj_key + ".weight"], loaded_model[down_proj_key + ".scale"], loaded_model[down_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.intermediate_size, module.intermediate_size, module.down_proj.weight.shape)
                loaded_model[down_proj_key + ".weight"] = down_proj_sliced_weight
                loaded_model[down_proj_key + ".scale"] = down_proj_sliced_scale
                loaded_model[down_proj_key + ".workspace"] = down_proj_sliced_workspace
            elif class_name in [model_helper.attn_w4a16_uniql_class_name]:
                attn_key = model_helper.get_attn_key(layer_idx)
                head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
                q_proj_key = f"{attn_key}.q_proj"
                q_proj_bias = None
                if q_proj_key + ".bias" in loaded_model.keys():
                    q_proj_bias = loaded_model[q_proj_key + ".bias"]
                q_proj_sliced_weight, q_proj_sliced_bias, q_proj_sliced_scale, q_proj_sliced_workspace = slice_w4a16_qk_proj(
                    loaded_model[q_proj_key + ".weight"], q_proj_bias,
                    loaded_model[q_proj_key + ".scale"], loaded_model[q_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.num_attention_heads, head_dim, module.head_dim, module.q_proj.weight.shape)
                loaded_model[q_proj_key + ".weight"] = q_proj_sliced_weight
                if q_proj_sliced_bias is not None:
                    loaded_model[q_proj_key + ".bias"] = q_proj_sliced_bias
                loaded_model[q_proj_key + ".scale"] = q_proj_sliced_scale
                loaded_model[q_proj_key + ".workspace"] = q_proj_sliced_workspace

                k_proj_key = f"{attn_key}.k_proj"
                k_proj_bias = None
                if k_proj_key + ".bias" in loaded_model.keys():
                    k_proj_bias = loaded_model[k_proj_key + ".bias"]
                k_proj_sliced_weight, k_proj_sliced_bias, k_proj_sliced_scale, k_proj_sliced_workspace = slice_w4a16_qk_proj(
                    loaded_model[k_proj_key + ".weight"], k_proj_bias,
                    loaded_model[k_proj_key + ".scale"], loaded_model[k_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.num_key_value_heads, head_dim, module.head_dim, module.k_proj.weight.shape)
                loaded_model[k_proj_key + ".weight"] = k_proj_sliced_weight
                if k_proj_sliced_bias is not None:
                    loaded_model[k_proj_key + ".bias"] = k_proj_sliced_bias
                loaded_model[k_proj_key + ".scale"] = k_proj_sliced_scale
                loaded_model[k_proj_key + ".workspace"] = k_proj_sliced_workspace

                v_proj_key = f"{attn_key}.v_proj"
                v_proj_sliced_weight, v_proj_sliced_scale, v_proj_sliced_workspace = slice_w4a16_v_proj(
                    loaded_model[v_proj_key + ".weight"], loaded_model[v_proj_key + ".scale"], loaded_model[v_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.num_key_value_heads, head_dim, module.head_dim, module.v_proj.weight.shape)
                loaded_model[v_proj_key + ".weight"] = v_proj_sliced_weight
                loaded_model[v_proj_key + ".scale"] = v_proj_sliced_scale
                loaded_model[v_proj_key + ".workspace"] = v_proj_sliced_workspace

                o_proj_key = f"{attn_key}.o_proj"
                o_proj_sliced_weight, o_proj_sliced_scale, o_proj_sliced_workspace = slice_w4a16_o_proj(
                    loaded_model[o_proj_key + ".weight"], loaded_model[o_proj_key + ".scale"], loaded_model[o_proj_key + ".workspace"],
                    model.config.hidden_size, model.config.num_attention_heads, head_dim, module.head_dim, module.o_proj.weight.shape)
                loaded_model[o_proj_key + ".weight"] = o_proj_sliced_weight
                loaded_model[o_proj_key + ".scale"] = o_proj_sliced_scale
                loaded_model[o_proj_key + ".workspace"] = o_proj_sliced_workspace

                indices_key = f"{attn_key}.indices"
                if indices_key in loaded_model.keys():
                    loaded_model[indices_key] = slice_indices(loaded_model[indices_key], head_dim, module.head_dim)
            elif class_name in [model_helper.mamba_w4a16_uniql_class_name]:
                mamba_key = model_helper.get_mamba_key(layer_idx)
                in_proj_key = f"{mamba_key}.in_proj"
                hidden_size = model.config.hidden_size
                intermediate_size = model_helper.get_mamba_full_intermediate_size
                ssm_state_size = model_helper.get_mamba_full_ssm_state_size
                in_proj_sliced_weight, in_proj_sliced_scale, in_proj_sliced_workspace = slice_w4a16_in_proj(
                    loaded_model[in_proj_key + ".weight"], loaded_model[in_proj_key + ".scale"], loaded_model[in_proj_key + ".workspace"],
                    hidden_size=hidden_size, n_groups=module.n_groups, num_heads=module.num_heads,
                    intermediate_size=intermediate_size, ssm_state_size=ssm_state_size,
                    load_intermediate_size=module.intermediate_size,
                    load_ssm_state_size=module.ssm_state_size,
                    load_shape=module.in_proj.weight.shape)
                loaded_model[in_proj_key + ".weight"] = in_proj_sliced_weight
                loaded_model[in_proj_key + ".scale"] = in_proj_sliced_scale
                loaded_model[in_proj_key + ".workspace"] = in_proj_sliced_workspace

                conv_key = f"{mamba_key}.conv1d"
                conv_sliced_weight, conv_sliced_bias = slice_conv1d(loaded_model[conv_key + ".weight"], loaded_model[conv_key + ".bias"],
                                                                    module.n_groups, module.num_heads, intermediate_size, ssm_state_size,
                                                                    module.intermediate_size, module.ssm_state_size)
                loaded_model[conv_key + ".weight"] = conv_sliced_weight
                loaded_model[conv_key + ".bias"] = conv_sliced_bias

                norm_key = f"{mamba_key}.norm"
                norm_sliced_weight = slice_norm(loaded_model[norm_key + ".weight"], module.n_groups, module.num_heads, intermediate_size, module.intermediate_size)
                loaded_model[norm_key + ".weight"] = norm_sliced_weight

                out_proj_key = f"{mamba_key}.out_proj"
                out_proj_sliced_weight, out_proj_sliced_scale, out_proj_sliced_workspace = slice_w4a16_out_proj(
                    loaded_model[out_proj_key + ".weight"], loaded_model[out_proj_key + ".scale"], loaded_model[out_proj_key + ".workspace"],
                    hidden_size=hidden_size, n_groups=module.n_groups, num_heads=module.num_heads,
                    intermediate_size=intermediate_size,
                    load_intermediate_size=module.intermediate_size,
                    load_shape=module.out_proj.weight.shape)
                loaded_model[out_proj_key + ".weight"] = out_proj_sliced_weight
                loaded_model[out_proj_key + ".scale"] = out_proj_sliced_scale
                loaded_model[out_proj_key + ".workspace"] = out_proj_sliced_workspace
            else:
                pass
    return loaded_model


def slice_w4a16_up_gate_proj(weight, scale, workspace, hidden_size, intermediate_size,
                       reduced_intermediate_size, load_shape, num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    pad_k = 0 # Din padding
    if hidden_size % group_size != 0:
        pad_k = group_size - hidden_size % group_size
    pad_n = 0 # Dout padding
    if intermediate_size % 256 != 0:
        pad_n = 256 - intermediate_size % 256
    reduced_pad_n = 0 # reduced Dout padding
    if reduced_intermediate_size % 256 != 0:
        reduced_pad_n = 256 - reduced_intermediate_size % 256

    """slice quantized weight"""
    assert load_shape[0] == weight.shape[0], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, hidden_size + pad_k, intermediate_size + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == (hidden_size, intermediate_size + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[1] >= reduced_intermediate_size + reduced_pad_n, "Reduced size shape exceeds recovered weight"
    loaded_wq = wq_full_recovered[:, :reduced_intermediate_size + reduced_pad_n] # [Din, Dout] = [k, n], where wq_full_recovered k is padded
    # permute and pack
    marlin_qqq_loaded_wq = get_w4a16_permute_weights(loaded_wq, hidden_size + pad_k, reduced_intermediate_size + reduced_pad_n,
                                                        num_bits, weight_perm, group_size)
    marlin_qqq_loaded_wq = marlin_qqq_loaded_wq.contiguous()

    """slice scale"""
    # we reversed the permutation of the scaling factors first
    scale_full = scale
    scale_full_reversed = w4a16_unpermute_scales(scale_full, hidden_size + pad_k, intermediate_size + pad_n, group_size)
    # then, cut to the target size and zero out the rest
    loaded_scale = scale_full_reversed[:, :reduced_intermediate_size + reduced_pad_n].contiguous() # load_shape[1] is padded
    loaded_scale[:, reduced_intermediate_size:] = 0 # zero out
    # finally, permute the scaling factors
    loaded_scale = w4a16_permute_scales(loaded_scale, hidden_size + pad_k, reduced_intermediate_size + reduced_pad_n, group_size)
    loaded_scale = loaded_scale.contiguous()

    """slice workspace"""
    workspace_full = workspace
    reduced_size_n = reduced_intermediate_size + reduced_pad_n
    reduced_max_workspace_size = ((reduced_size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
    loaded_workspace = workspace_full[:reduced_max_workspace_size].contiguous()

    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace


def slice_w4a16_down_proj(weight, scale, workspace, hidden_size, intermediate_size,
                    reduced_intermediate_size, load_shape, num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"
    pad_k = 0 # Din padding
    if intermediate_size % group_size != 0:
        pad_k = group_size - intermediate_size % group_size
    reduced_pad_k = 0 # reduced Din padding
    if reduced_intermediate_size % group_size != 0:
        reduced_pad_k = group_size - reduced_intermediate_size % group_size
    pad_n = 0 # Dout padding
    if hidden_size % 256 != 0:
        pad_n = 256 - hidden_size % 256

    assert load_shape[1] == weight.shape[1], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, intermediate_size + pad_k, hidden_size + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == (intermediate_size + pad_k, hidden_size + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[0] >= reduced_intermediate_size + reduced_pad_k, "Reduced size shape exceeds recovered weight"
    loaded_wq = wq_full_recovered[:reduced_intermediate_size + reduced_pad_k, :] # [Din, Dout] = [k, n], where wq_full_recovered k is padded
    # permute and pack
    marlin_qqq_loaded_wq = get_w4a16_permute_weights(loaded_wq, reduced_intermediate_size + reduced_pad_k, hidden_size + pad_n,
                                                        num_bits, weight_perm, group_size)
    marlin_qqq_loaded_wq = marlin_qqq_loaded_wq.contiguous()

    """slice scale"""
    scale_full = scale
    load_ngroups = int(math.ceil(reduced_intermediate_size / 128))
    loaded_scale = scale_full[:load_ngroups, :]
    loaded_scale = loaded_scale.contiguous()

    """slice workspace"""
    # down_proj do nothing to size_n
    loaded_workspace = workspace
    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace


def slice_w4a16_qk_proj(weight, bias, scale, workspace, hidden_size, nheads, head_dim, reduced_head_dim,
                        load_shape, num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    pad_k = 0 # Din padding
    if hidden_size % group_size != 0:
        pad_k = group_size - hidden_size % group_size
    pad_n = 0 # Dout padding
    if (nheads * head_dim) % 256 != 0:
        pad_n = 256 - (nheads * head_dim) % 256
    reduced_pad_n = 0 # reduced Dout padding
    if (nheads * reduced_head_dim) % 256 != 0:
        reduced_pad_n = 256 - (nheads * reduced_head_dim) % 256

    """slice quantized weight"""
    assert load_shape[0] == weight.shape[0], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, hidden_size + pad_k, (nheads * head_dim) + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == (hidden_size, (nheads * head_dim) + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[1] >= (nheads * reduced_head_dim) + reduced_pad_n, "Reduced size shape exceeds recovered weight"
    if pad_k > 0: # remove the Din padding
        wq_full_recovered = wq_full_recovered[:hidden_size, :]
    if pad_n > 0: # remove the Dout padding
        wq_full_recovered = wq_full_recovered[:, :(nheads * head_dim)]

    wq_full_recovered = rearrange(wq_full_recovered, 'h (n d)-> h n d', d=head_dim)  # [Din Dout] -> [h, n, d], h: hidden_size
    half = head_dim // 2
    hd_half = reduced_head_dim // 2
    loaded_wq = torch.cat([wq_full_recovered[:, :, :hd_half],
                            wq_full_recovered[:, :, half:half+hd_half]], dim=-1)
    loaded_wq = rearrange(loaded_wq, 'h n d -> h (n d)', d=reduced_head_dim).contiguous()
    # pad
    if pad_k > 0: # pad the Din
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, 0, 0, pad_k), "constant", 0)  # pad rows (dim 0) with zeros at the bottom
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_wq: [Din, Dout]
    # permute and pack
    marlin_qqq_loaded_wq = get_w4a16_permute_weights(loaded_wq, hidden_size + pad_k, (nheads * reduced_head_dim) + reduced_pad_n,
                                                        num_bits, weight_perm, group_size)
    marlin_qqq_loaded_wq = marlin_qqq_loaded_wq.contiguous()

    """slice scale"""
    scale_full = scale
    # we reversed the permutation of the scaling factors first
    scale_full_reversed = w4a16_unpermute_scales(scale_full, hidden_size + pad_k, (nheads * head_dim) + pad_n, group_size)
    if pad_n > 0: # remove the Dout padding
        scale_full_reversed = scale_full_reversed[:, :(nheads * head_dim)]
    scale_full_reversed = rearrange(scale_full_reversed, 'g (n d)-> g n d', d=head_dim)  # [Din Dout] -> [h, n, d], h: hidden_size
    half = head_dim // 2
    hd_half = reduced_head_dim // 2
    loaded_scale = torch.cat([scale_full_reversed[:, :, :hd_half],
                            scale_full_reversed[:, :, half:half+hd_half]], dim=-1)
    loaded_scale = rearrange(loaded_scale, 'g n d -> g (n d)', d=reduced_head_dim).contiguous()
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_scale = torch.nn.functional.pad(loaded_scale, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_scale: [Din, Dout]
    # finally, permute the scaling factors
    loaded_scale = w4a16_permute_scales(loaded_scale, hidden_size + pad_k, (nheads * reduced_head_dim) + reduced_pad_n, group_size)
    loaded_scale = loaded_scale.contiguous()

    """slice workspace"""
    workspace_full = workspace
    reduced_size_n = (nheads * reduced_head_dim) + reduced_pad_n
    reduced_max_workspace_size = ((reduced_size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
    loaded_workspace = workspace_full[:reduced_max_workspace_size].contiguous()

    """slice bias"""
    if bias is not None:
        bias = rearrange(bias, '(n d) -> n d', d=head_dim)
        half = head_dim // 2
        hd_half = reduced_head_dim // 2
        loaded_bias = torch.cat([bias[:, :hd_half], bias[:, half:half+hd_half]], dim=1)
        loaded_bias = rearrange(loaded_bias, 'n d -> (n d)')
    else:
        loaded_bias = None

    return marlin_qqq_loaded_wq, loaded_bias, loaded_scale, loaded_workspace


def slice_w4a16_v_proj(weight, scale, workspace, hidden_size, nheads, head_dim, reduced_head_dim,
                        load_shape, num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    pad_k = 0 # Din padding
    if hidden_size % group_size != 0:
        pad_k = group_size - hidden_size % group_size
    pad_n = 0 # Dout padding
    if (nheads * head_dim) % 256 != 0:
        pad_n = 256 - (nheads * head_dim) % 256
    reduced_pad_n = 0 # reduced Dout padding
    if (nheads * reduced_head_dim) % 256 != 0:
        reduced_pad_n = 256 - (nheads * reduced_head_dim) % 256

    """slice quantized weight"""
    assert load_shape[0] == weight.shape[0], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, hidden_size + pad_k, (nheads * head_dim) + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == (hidden_size, (nheads * head_dim) + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[1] >= (nheads * reduced_head_dim) + reduced_pad_n, "Reduced size shape exceeds recovered weight"
    if pad_k > 0: # remove the Din padding
        wq_full_recovered = wq_full_recovered[:hidden_size, :]
    if pad_n > 0: # remove the Dout padding
        wq_full_recovered = wq_full_recovered[:, :(nheads * head_dim)]

    wq_full_recovered = rearrange(wq_full_recovered, 'h (n d)-> h n d', d=head_dim)  # [Din Dout] -> [h, n, d], h: hidden_size
    loaded_wq = wq_full_recovered[:, :, :reduced_head_dim]
    loaded_wq = rearrange(loaded_wq, 'h n d -> h (n d)', d=reduced_head_dim).contiguous()
    # pad
    if pad_k > 0: # pad the Din
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, 0, 0, pad_k), "constant", 0)  # pad rows (dim 0) with zeros at the bottom
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_wq: [Din, Dout]
    # permute and pack
    marlin_qqq_loaded_wq = get_w4a16_permute_weights(loaded_wq, hidden_size + pad_k, (nheads * reduced_head_dim) + reduced_pad_n,
                                                        num_bits, weight_perm, group_size)
    marlin_qqq_loaded_wq = marlin_qqq_loaded_wq.contiguous()

    """slice scale"""
    scale_full = scale
    # we reversed the permutation of the scaling factors first
    scale_full_reversed = w4a16_unpermute_scales(scale_full, hidden_size + pad_k, (nheads * head_dim) + pad_n, group_size)
    if pad_n > 0: # remove the Dout padding
        scale_full_reversed = scale_full_reversed[:, :(nheads * head_dim)]
    scale_full_reversed = rearrange(scale_full_reversed, 'g (n d)-> g n d', d=head_dim)  # [nGroup Dout] -> [g n d], h: hidden_size
    loaded_scale = scale_full_reversed[:, :, :reduced_head_dim]
    loaded_scale = rearrange(loaded_scale, 'g n d -> g (n d)', d=reduced_head_dim).contiguous()
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_scale = torch.nn.functional.pad(loaded_scale, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_scale: [Din, Dout]
    # finally, permute the scaling factors
    loaded_scale = w4a16_permute_scales(loaded_scale, hidden_size + pad_k, (nheads * reduced_head_dim) + reduced_pad_n, group_size)
    loaded_scale = loaded_scale.contiguous()

    """slice workspace"""
    workspace_full = workspace
    reduced_size_n = (nheads * reduced_head_dim) + reduced_pad_n
    reduced_max_workspace_size = ((reduced_size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
    loaded_workspace = workspace_full[:reduced_max_workspace_size].contiguous()

    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace    



def slice_w4a16_o_proj(weight, scale, workspace, hidden_size, nheads, head_dim, reduced_head_dim,
                        load_shape, num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    pad_k = 0 # Din padding
    if (nheads * head_dim) % group_size != 0:
        pad_k = group_size - (nheads * head_dim) % group_size
    reduced_pad_k = 0 # reduced Din padding
    if (nheads * reduced_head_dim) % group_size != 0:
        reduced_pad_k = group_size - (nheads * reduced_head_dim) % group_size
    pad_n = 0 # Dout padding
    if hidden_size % 256 != 0:
        pad_n = 256 - hidden_size % 256

    """slice quantized weight"""
    assert load_shape[1] == weight.shape[1], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack the weight
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, (nheads * head_dim) + pad_k, hidden_size + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == ((nheads * head_dim) + pad_k, hidden_size + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[0] >= (nheads * reduced_head_dim) + reduced_pad_k, "Reduced size shape exceeds recovered weight"
    # unpermute the scale
    scale_full = scale
    scale_full_reversed = w4a16_unpermute_scales(scale_full, (nheads * head_dim) + pad_k, hidden_size + pad_n, group_size)
    # remove the Dout padding
    if pad_n > 0: 
        wq_full_recovered = wq_full_recovered[:, :hidden_size]
        scale_full_reversed = scale_full_reversed[:, :hidden_size] # [nGroup, Dout]

    # dequantize by scale
    dtype = scale_full_reversed.dtype
    ngroup = scale_full_reversed.shape[0]
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2
    wq_full_recovered = wq_full_recovered.to(dtype) - half_q_val
    wq_full_recovered = rearrange(wq_full_recovered, '(g l) h -> g l h', g=ngroup, l=group_size)  # [Din Dout] -> [g l h], g: ngroup, h: hidden_size
    w_full_recovered = wq_full_recovered * scale_full_reversed.unsqueeze(1)
    w_full_recovered = w_full_recovered.reshape(-1, hidden_size)
    if pad_k > 0: # remove the Din padding
        w_full_recovered = w_full_recovered[:(nheads * head_dim), :]

    # slice the weight
    w_full_recovered = rearrange(w_full_recovered, '(n d) h -> n d h', d=head_dim)  # [Din Dout] -> [n d h], h: hidden_size
    loaded_wq = w_full_recovered[:, :reduced_head_dim, :]
    loaded_wq = rearrange(loaded_wq, 'n d h -> (n d) h', d=reduced_head_dim).contiguous()

    # re-compute scaling factors,quantize, permute, and pack 
    _, marlin_qqq_loaded_wq, loaded_scale = w4a16_quantize(loaded_wq, num_bits, group_size, scale=None, pad_out=pad_n)

    """slice workspace"""
    loaded_workspace = workspace

    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace


def slice_indices(indices, head_dim, reduced_head_dim):
    half = head_dim// 2
    hd_half = reduced_head_dim // 2
    loaded_indices = torch.cat([indices[:, :hd_half], indices[:, half:half+hd_half]], dim=-1)
    return loaded_indices.to(torch.uint8)

# for Mamba in_proj
def slice_w4a16_in_proj(weight, scale, workspace, hidden_size,
                        n_groups, num_heads, intermediate_size, ssm_state_size,
                        load_intermediate_size, load_ssm_state_size, load_shape,
                        num_bits = 4, group_size = 128):

    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    # output z, x, B, C, dt: 2 * intermediate_size + 2 * n_groups * ssm_state_size + num_heads
    projection_size = 2*intermediate_size + 2*n_groups*ssm_state_size + num_heads
    pad_k = 0 # Din padding
    if hidden_size % group_size != 0:
        pad_k = group_size - hidden_size % group_size
    pad_n = 0 # Dout padding
    if projection_size % 256 != 0:
        pad_n = 256 - projection_size % 256

    reduced_projection_size = 2*load_intermediate_size + 2*n_groups*load_ssm_state_size + num_heads
    reduced_pad_n = 0 # reduced Dout padding
    if reduced_projection_size % 256 != 0:
        reduced_pad_n = 256 - reduced_projection_size % 256

    """slice quantized weight"""
    assert load_shape[0] == weight.shape[0], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack, quantized weight has shape[Din, Dout]
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, hidden_size + pad_k, projection_size + pad_n, num_bits,
                                                     weight_perm, group_size)
    assert wq_full_recovered.shape == (hidden_size + pad_k, projection_size + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[1] >= reduced_projection_size + reduced_pad_n, "Reduced size shape exceeds recovered weight"
    if pad_k > 0: # remove the Din padding
        wq_full_recovered = wq_full_recovered[:hidden_size, :]
    if pad_n > 0: # remove the Dout padding
        wq_full_recovered = wq_full_recovered[:, :projection_size]

    # split the weight into z, x, B, C, dt
    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(weight.device)
    topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(weight.device)
    z_start = 0
    z_indices = z_start + topk_head_indices
    w_z = wq_full_recovered[:, z_indices] # [Din, Dout]
    x_start = z_start + intermediate_size
    x_indices = x_start + topk_head_indices
    w_x = wq_full_recovered[:, x_indices]
    b_start = x_start + intermediate_size
    b_indices = b_start + topk_ssm_state_indices
    w_b = wq_full_recovered[:, b_indices]
    c_start = b_start + n_groups*ssm_state_size
    c_indices = c_start + topk_ssm_state_indices
    w_c = wq_full_recovered[:, c_indices]
    dt_start = c_start + n_groups*ssm_state_size
    w_dt = wq_full_recovered[:, dt_start:]
    loaded_wq = torch.cat([w_z, w_x, w_b, w_c, w_dt], dim=1)
    # pad
    if pad_k > 0: # pad the Din
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, 0, 0, pad_k), "constant", 0)  # pad rows (dim 0) with zeros at the bottom
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_wq = torch.nn.functional.pad(loaded_wq, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_wq: [Din, Dout]
    # permute and pack
    marlin_qqq_loaded_wq = get_w4a16_permute_weights(loaded_wq, hidden_size + pad_k, reduced_projection_size + reduced_pad_n,
                                                     num_bits, weight_perm, group_size)
    marlin_qqq_loaded_wq = marlin_qqq_loaded_wq.contiguous()

    """slice scale"""
    scale_full = scale
    # we reversed the permutation of the scaling factors first
    scale_full_reversed = w4a16_unpermute_scales(scale_full, hidden_size + pad_k, projection_size + pad_n, group_size)
    if pad_n > 0: # remove the Dout padding
        scale_full_reversed = scale_full_reversed[:, :projection_size]
    # scale_full_reversed = rearrange(scale_full_reversed, 'g d-> g d', d=projection_size)  # [Din Dout] -> [h, n, d], h: hidden_size
    topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(scale_full.device)
    topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(scale_full.device)
    z_start = 0
    z_indices = z_start + topk_head_indices
    scale_z = scale_full_reversed[:, z_indices]
    x_start = z_start + intermediate_size
    x_indices = x_start + topk_head_indices
    scale_x = scale_full_reversed[:, x_indices]
    b_start = x_start + intermediate_size
    b_indices = b_start + topk_ssm_state_indices
    scale_b = scale_full_reversed[:, b_indices]
    c_start = b_start + n_groups*ssm_state_size
    c_indices = c_start + topk_ssm_state_indices
    scale_c = scale_full_reversed[:, c_indices]
    dt_start = c_start + n_groups*ssm_state_size
    scale_dt = scale_full_reversed[:, dt_start:]
    loaded_scale = torch.cat([scale_z, scale_x, scale_b, scale_c, scale_dt], dim=-1).contiguous()
    if reduced_pad_n > 0: # pad the reduced Dout
        loaded_scale = torch.nn.functional.pad(loaded_scale, (0, reduced_pad_n, 0, 0), "constant", 0) # loaded_scale: [Din, Dout]
    # finally, permute the scaling factors
    loaded_scale = w4a16_permute_scales(loaded_scale, hidden_size + pad_k, reduced_projection_size + reduced_pad_n, group_size)
    loaded_scale = loaded_scale.contiguous()

    """slice workspace"""
    workspace_full = workspace
    reduced_size_n = reduced_projection_size + reduced_pad_n
    reduced_max_workspace_size = ((reduced_size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
    loaded_workspace = workspace_full[:reduced_max_workspace_size].contiguous()

    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace


def slice_conv1d(weight, bias, n_groups, num_heads, intermediate_size, ssm_state_size,
                 load_intermediate_size, load_ssm_state_size):

    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(weight.device)
    topk_ssm_state_indices = get_topk_ssm_state_indices(n_groups, ssm_state_size, load_ssm_state_size).to(weight.device)

    # conv_wx
    x_start = 0
    x_indices = x_start + topk_head_indices
    conv_w_x = weight[x_indices, :, :]
    conv_bias_x = bias[x_indices]
    # conv_wb
    b_start = x_start + intermediate_size
    b_indices = b_start + topk_ssm_state_indices
    conv_w_b = weight[b_indices, :, :]
    conv_bias_b = bias[b_indices]
    # conv_wc
    c_start = b_start + n_groups*ssm_state_size
    c_indices = c_start + topk_ssm_state_indices
    conv_w_c = weight[c_indices, :, :]
    conv_bias_c = bias[c_indices]
    
    # concat
    loaded_conv_weight = torch.cat([conv_w_x, conv_w_b, conv_w_c], dim=0)
    loaded_conv_bias = torch.cat([conv_bias_x, conv_bias_b, conv_bias_c], dim=0)
    return loaded_conv_weight.contiguous(), loaded_conv_bias.contiguous()


def slice_norm(weight, n_groups, num_heads, intermediate_size, load_intermediate_size):

    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(weight.device)
    loaded_weight = weight[topk_head_indices]
    return loaded_weight.contiguous()


def slice_w4a16_out_proj(weight, scale, workspace, hidden_size,
                        n_groups, num_heads, intermediate_size,
                        load_intermediate_size, load_shape,
                        num_bits = 4, group_size = 128):
    assert num_bits == 4, "Only support 4-bit quantization for now"
    assert group_size in [-1, 128], "Only support group_size of -1 or 128 for now"

    pad_k = 0 # Din padding
    if intermediate_size % group_size != 0:
        pad_k = group_size - intermediate_size % group_size
    reduced_pad_k = 0 # reduced Din padding
    if load_intermediate_size % group_size != 0:
        reduced_pad_k = group_size - load_intermediate_size % group_size
    pad_n = 0 # Dout padding
    if hidden_size % 256 != 0:
        pad_n = 256 - hidden_size % 256

    """slice quantized weight"""
    assert load_shape[1] == weight.shape[1], "hidden_size dimension mismatch" # the 4-bit packed [Din, Dout]
    wq_full = weight
    weight_perm = get_w4a16_weight_perms()
    # unpermute and unpack the weight
    wq_full_recovered = unpack_w4a16_permute_weights(wq_full, intermediate_size + pad_k, hidden_size + pad_n, num_bits,
                                                        weight_perm, group_size)
    assert wq_full_recovered.shape == (intermediate_size + pad_k, hidden_size + pad_n), "Recovered weight shape mismatch"
    assert wq_full_recovered.shape[0] >= (load_intermediate_size + reduced_pad_k), "Reduced size shape exceeds recovered weight"
    # unpermute the scale
    scale_full = scale
    scale_full_reversed = w4a16_unpermute_scales(scale_full, intermediate_size + pad_k, hidden_size + pad_n, group_size)
    # remove the Dout padding
    if pad_n > 0: 
        wq_full_recovered = wq_full_recovered[:, :hidden_size]
        scale_full_reversed = scale_full_reversed[:, :hidden_size] # [nGroup, Dout]

    # dequantize by scale
    dtype = scale_full_reversed.dtype
    scale_ngroup = scale_full_reversed.shape[0]
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2
    wq_full_recovered = wq_full_recovered.to(dtype) - half_q_val
    wq_full_recovered = rearrange(wq_full_recovered, '(g l) h -> g l h', g=scale_ngroup, l=group_size)  # [Din Dout] -> [g l h], g: ngroup, h: hidden_size
    w_full_recovered = wq_full_recovered * scale_full_reversed.unsqueeze(1)
    w_full_recovered = w_full_recovered.reshape(-1, hidden_size)
    if pad_k > 0: # remove the Din padding
        w_full_recovered = w_full_recovered[:intermediate_size, :]

    # slice the weight
    head_dim = intermediate_size // num_heads
    topk_head_dim = load_intermediate_size // num_heads
    topk_head_indices = get_topk_head_indices(n_groups, num_heads, head_dim, topk_head_dim).to(w_full_recovered.device)
    loaded_weight = w_full_recovered[topk_head_indices, :]

    # re-compute scaling factors,quantize, permute, and pack 
    _, marlin_qqq_loaded_wq, loaded_scale = w4a16_quantize(loaded_weight, num_bits, group_size, scale=None, pad_out=pad_n)

    """slice workspace"""
    loaded_workspace = workspace

    return marlin_qqq_loaded_wq, loaded_scale, loaded_workspace