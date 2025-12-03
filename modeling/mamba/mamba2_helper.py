import copy
import torch
import torch.nn as nn

from modeling.module_helpers import get_size
# from modeling.bamba.bamba_blocks import apply_rotary_pos_emb
from modeling.mamba.mamba2_blocks import Mamba2Cache
from modeling.mamba.mamba2_blocks import MambaRMSNormGated
from modeling.mamba.mamba2_simple_blocks import Mamba2MixerSimple
from modeling.mamba.mamba2_uniql_blocks import Mamba2UniQLMixer
from modeling.mamba.mamba2_qblocks import W4A16Mamba2Mixer
from modeling.mamba.mamba2_uniql_qblocks import W4A16Mamba2UniQLMixer

from compress.compress_helpers import ssd_BC_compression
from compress.compress_helpers import ssd_xo_compression
from compress.compress_helpers import ssd_state_aware_xo_compression

class Mamba2Helper():

    def __init__(self, model_config):
        self.config = copy.deepcopy(model_config)
        self.module_mapping = {
            "Mamba2Mixer": "ssm",
            "Mamba2MixerSimple": "ssm",
            "Mamba2UniQLMixer": "ssm",
            "W4A16Mamba2Mixer": "ssm",
            "W4A16Mamba2UniQLMixer": "ssm",
            "MambaRMSNormGated": "norm",
            "Mamba2RMSNorm": "norm",
        }

    def get_layer_size(self, model):
        module_size = {module_type: 0 for class_name, module_type in self.module_mapping.items()}
        for layer_idx, layer in enumerate(model.backbone.layers):
            # only iterate the direct children of the layer
            for name, module in layer.named_children():
                class_name = module.__class__.__name__
                module_type = self.module_mapping[class_name]
                if module_type == "moe":
                    if module.num_experts > 1:
                        module_size["moe"] += get_size(module)
                    else:
                        module_size["mlp"] += get_size(module)
                else:
                    module_size[module_type] += get_size(module)
        module_size["norm"] += get_size(model.backbone.norm_f)
        module_size["embedding"] = get_size(model.backbone.embeddings)
        module_size["output"] = 0 if model.lm_head.weight is model.backbone.embeddings.weight else get_size(model.lm_head)
        return module_size
    
    def get_cache_size(self, batch_size, prompt_len):
        cache = Mamba2Cache(self.config, batch_size, dtype=torch.float16, device=None)
        conv_state_size = 0
        ssm_state_size = 0
        for (conv_state, ssm_state) in zip(cache.conv_states, cache.ssm_states):
            conv_state_size += conv_state.nelement() * conv_state.element_size()
            ssm_state_size += ssm_state.nelement() * ssm_state.element_size()
        kv_cache_size = 0
        return conv_state_size, ssm_state_size, kv_cache_size

    def get_embeddings(self, model):
        return model.backbone.embeddings

    def set_embeddings(self, model, new_embedding):
        model.backbone.embeddings = new_embedding

    def get_layers(self, model):
        return model.backbone.layers

    @property
    def layer_step(self):
        # a layer means a mixer block plus a mlp block in Bamba
        return 1

    def get_input_layernorm(self, layer):
        return layer.norm

    def get_mlp_norm(self, layer):
        return None
    
    def get_mlp(self, layer):
        return None

    def get_mlp_key(self, layer_idx):
        return None

    def get_mamba_norm(self, layer):
        return layer.norm
    
    def get_mamba(self, layer):
        return layer.mixer

    def get_mamba_key(self, layer_idx):
        return f"backbone.layers.{layer_idx}.mixer"

    def get_attn_norm(self, layer):
        return None
    
    def get_attn(self, layer):
        return None

    def get_attn_key(self, layer_idx):
        return None

    def get_final_layernorm(self, model):
        return model.backbone.norm_f
    
    def get_lm_head(self, model):
        return model.lm_head

    def set_lm_head(self, model, new_head):
        model.lm_head = new_head

    @property
    def mlp_class_name(self):
        return None

    @property
    def mlp_simple_class_name(self):
        return None

    @property
    def mlp_uniql_class_name(self):
        return None

    @property
    def mlp_w4a16_uniql_class_name(self):
        return None

    @property
    def get_mamba_full_intermediate_size(self):
        return int(self.config.expand * self.config.hidden_size)

    @property
    def get_mamba_full_ssm_state_size(self):
        return self.config.state_size

    def compress_mlp(self, layer, down_x, ratio):
        raise NotImplementedError("compress_mlp is not implemented for Mamba2")
 

    def replace_simple_mlp(self, layer, use_had_transform=False):
        raise NotImplementedError("replace_simple_mlp is not implemented for Mamba2")
    
    def replace_w4a16_mlp(self, layer):
        raise NotImplementedError("replace_w4a16_mlp is not implemented for Mamba2")

    @property
    def layer_mlp_ops(self):
        return None

    @property
    def mamba_class_name(self):
        return "Mamba2Mixer"

    @property
    def mamba_simple_class_name(self):
        return "Mamba2MixerSimple"

    @property
    def mamba_uniql_class_name(self):
        return "Mamba2UniQLMixer"

    @property
    def mamba_w4a16_uniql_class_name(self):
        return "W4A16Mamba2UniQLMixer"
    
    def compress_mamba(self, layer, hidden_states, y_normed, ratio):

        # Compress B and C proj
        reduced_dstate = 16 * int(round(layer.mixer.ssm_state_size * ratio / 16.))
        B_proj, C_proj, compressed_conv1d = ssd_BC_compression(
            layer.mixer.B_proj, layer.mixer.C_proj, layer.mixer.dt_proj, hidden_states, reduced_dstate,
            layer.mixer.intermediate_size, layer.mixer.num_heads, layer.mixer.n_groups, layer.mixer.ssm_state_size, layer.mixer.conv1d)

        # Compress z, x, and out proj
        head_dim = layer.mixer.intermediate_size // layer.mixer.num_heads
        reduced_head_dim = 2 * int(round(head_dim * ratio / 2.))
        reduced_intermediate_size = reduced_head_dim * layer.mixer.num_heads
        # compressed_gated_norm = MambaRMSNormGated(layer.mixer.intermediate_size,
        #                                           eps=layer.mixer.norm.variance_epsilon,
        #                                           group_size=layer.mixer.intermediate_size // layer.mixer.n_groups)
        compressed_gated_norm = MambaRMSNormGated(reduced_intermediate_size,
                                                  eps=layer.mixer.norm.eps,
                                                  group_size=reduced_intermediate_size // layer.mixer.n_groups,
                                                  norm_before_gate=layer.mixer.norm.norm_before_gate)
        x_proj, out_proj, z_proj, compressed_gated_norm, compressed_conv1d = ssd_state_aware_xo_compression(
                       layer.mixer.x_proj, layer.mixer.out_proj, layer.mixer.z_proj,
                       hidden_states, layer.mixer.B_proj, layer.mixer.dt_proj,
                       reduced_head_dim, layer.mixer.intermediate_size,
                       layer.mixer.n_groups,layer.mixer.num_heads, layer.mixer.head_dim,
                       layer.mixer.norm, compressed_gated_norm, layer.mixer.conv1d, compressed_conv1d,
                       layer.mixer.A_log, layer.mixer.chunk_size, layer.mixer.dt_bias,
                       dt_softplus=True, dt_limit=(0.0, float("inf")), seq_idx=None, ridge_lambda=1e-4)

        compressed_mamba = Mamba2UniQLMixer(layer.mixer.config, layer.mixer.layer_idx, flex_layer_ratio={"mamba_ratio": ratio})
        in_proj_weight = torch.cat([z_proj.weight.data, x_proj.weight.data,
                                    B_proj.weight.data, C_proj.weight.data,
                                    layer.mixer.dt_proj.weight.data], dim=0)
        compressed_mamba.in_proj.weight.data = in_proj_weight
        compressed_mamba.out_proj = out_proj
        compressed_mamba.norm = compressed_gated_norm
        compressed_mamba.conv1d = compressed_conv1d
        compressed_mamba.D = layer.mixer.D
        compressed_mamba.dt_bias = layer.mixer.dt_bias
        compressed_mamba.A_log = layer.mixer.A_log
        layer.mixer = compressed_mamba

        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_mamba_d_state" not in self.config.compress_config:
            self.config.compress_config["reduced_mamba_d_state"] = {}
        if "reduced_mamba_d_head" not in self.config.compress_config:
            self.config.compress_config["reduced_mamba_d_head"] = {}
        self.config.compress_config["reduced_mamba_d_state"][layer.mixer.layer_idx] = reduced_dstate
        self.config.compress_config["reduced_mamba_d_head"][layer.mixer.layer_idx] = reduced_head_dim


    def replace_simple_mamba(self, layer, use_had_transform=False):
        layer.mixer = Mamba2MixerSimple(layer.mixer, use_had_transform=use_had_transform)

    def replace_w4a16_mamba(self, layer):
        if layer.mixer.__class__.__name__ == "Mamba2Mixer":
            layer.mixer = W4A16Mamba2Mixer.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "Mamba2UniQLMixer":
            layer.mixer = W4A16Mamba2UniQLMixer.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "Mamba2MixerSimple":
            layer.mixer.fuse_in_proj()
            # layer.mixer = W4A16Mamba2Mixer.from_fp16(layer.mixer)
            if hasattr(layer.mixer.config, "compress_config"):
                # Mamba2UniQLMixer -> Mamba2MixerSimple use this
                layer.mixer = W4A16Mamba2UniQLMixer.from_fp16(layer.mixer)
            else:
                # Mamba2Mixer -> Mamba2MixerSimple use this
                layer.mixer = W4A16Mamba2Mixer.from_fp16(layer.mixer)
        else:
            raise ValueError(f"Unsupported Mamba class: {layer.mixer.__class__.__name__}")

    @property
    def layer_mamba_ops(self):
        return ["mixer.out_proj", "mixer.z_proj", "mixer.x_proj", "mixer.B_proj", "mixer.C_proj", "mixer.dt_proj"]

    @property
    def attn_class_name(self):
        return None

    @property
    def attn_simple_class_name(self):
        return None

    @property
    def attn_uniql_class_name(self):
        return None

    @property
    def attn_w4a16_uniql_class_name(self):
        return None

    def compress_attn(self, layer, qkv_x, ratio, position_embedding):
        raise NotImplementedError("compress_attn is not implemented for Mamba2")

    def replace_simple_attn(self, layer, use_had_transform=False):
        raise NotImplementedError("replace_simple_attn is not implemented for Mamba2")

    def replace_w4a16_attn(self, layer):
        raise NotImplementedError("replace_w4a16_attn is not implemented for Mamba2")

    @property
    def layer_attn_ops(self):
        return None
