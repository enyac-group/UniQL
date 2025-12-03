import logging
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from modeling.module_helpers import get_size

## nemotron_h
from modeling.nemotron_h.modeling_nemotron_h import HybridMambaAttentionDynamicCache as NemotronHCache
from modeling.nemotron_h.nemotron_h_blocks import (
    MambaRMSNormGated,
)
from modeling.nemotron_h.nemotron_h_simple_blocks import (
    NemotronHMamba2MixerSimple,
    NemotronHAttentionSimple,
    NemotronHMLPSimple,
)
from modeling.nemotron_h.nemotron_h_qblocks import (
    W4A16NemotronHAttention,
    W4A16NemotronHMLP,
    W4A16NemotronHMamba2Mixer
)
from modeling.nemotron_h.nemotron_h_uniql_blocks import (
    NemotronHUniQLMamba2Mixer,
    NemotronHUniQLMLP,
    NemotronHUniQLAttention
)
from modeling.nemotron_h.nemotron_h_uniql_qblocks import (
    W4A16NemotronHUniQLMamba2Mixer,
    W4A16NemotronHUniQLMLP,
    W4A16NemotronHUniQLAttention
)

from compress.compress_helpers import mlp_compression
from compress.compress_helpers import ssd_BC_compression, ssd_state_aware_xo_compression
from compress.compress_helpers import attn_qk_compression_gqa, attn_vo_compression_gqa

class NemotronHelper():

    def __init__(self, model_config):
        self.config = model_config
        self.module_mapping = {
            "NemotronHMamba2Mixer": "ssm",
            "W4A16NemotronHMamba2Mixer": "ssm",
            "NemotronHMamba2MixerSimple": "ssm",
            "NemotronHUniQLMamba2Mixer": "ssm",
            "W4A16NemotronHUniQLMamba2Mixer": "ssm",
            "NemotronHMLP": "mlp",
            "W4A16NemotronHMLP": "mlp",
            "NemotronHMLPSimple": "mlp",
            "NemotronHUniQLMLP": "mlp",
            "W4A16NemotronHUniQLMLP": "mlp",
            "NemotronHAttention": "attn",
            "W4A16NemotronHAttention": "attn",
            "NemotronHAttentionSimple": "attn",
            "NemotronHUniQLAttention": "attn",
            "W4A16NemotronHUniQLAttention": "attn",
            "NemotronHFlashAttention2": "attn",
            "NemotronHSdpaAttention": "attn",
            "MambaRMSNormGated": "norm",
            "NemotronHRMSNorm": "norm",
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
        # module_size["others"] = get_size(model.model.memory_tokens)
        return module_size
    
    def get_cache_size(self, batch_size, prompt_len):
        cache = NemotronHCache(self.config, batch_size, dtype=torch.float16, device=None)
        conv_state_size = 0
        ssm_state_size = 0
        for (conv_state, ssm_state) in zip(cache.conv_states, cache.ssm_states):
            conv_state_size += conv_state.nelement() * conv_state.element_size()
            ssm_state_size += ssm_state.nelement() * ssm_state.element_size()
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        kv_cache_size = 0
        kv_dim = head_dim * self.config.num_key_value_heads
        for attn_layer in cache.transformer_layers:
            kv_cache_size += batch_size * prompt_len * kv_dim * torch.finfo(cache.dtype).bits // 8 * 2 # K and V
        return conv_state_size, ssm_state_size, kv_cache_size

    def get_embeddings(self, model):
        return model.backbone.embeddings

    def set_embeddings(self, model, new_embedding):
        model.backbone.embeddings = new_embedding

    def get_layers(self, model):
        return model.backbone.layers

    @property
    def layer_step(self):
        # a layer means a mixer block plus a mlp block in Nemotron-H
        return 2

    def get_input_layernorm(self, layer):
        return layer.norm
    
    def get_mlp_norm(self, layer):
        return layer.norm
    
    def get_mlp(self, layer):
        return layer.mixer

    def get_mlp_key(self, layer_idx):
        return f"backbone.layers.{layer_idx}.mixer"

    def get_mamba_norm(self, layer):
        return layer.norm
    
    def get_mamba(self, layer):
        return layer.mixer

    def get_mamba_key(self, layer_idx):
        return f"backbone.layers.{layer_idx}.mixer"

    def get_attn_norm(self, layer):
        return layer.norm
    
    def get_attn(self, layer):
        return layer.mixer

    def get_attn_key(self, layer_idx):
        return f"backbone.layers.{layer_idx}.mixer"

    def get_final_layernorm(self, model):
        return model.backbone.norm_f
    
    def get_lm_head(self, model):
        return model.lm_head
    
    def set_lm_head(self, model, new_head):
        model.lm_head = new_head

    @property
    def mlp_class_name(self):
        return "NemotronHMLP"

    @property
    def mlp_simple_class_name(self):
        return "NemotronHMLPSimple"
    
    @property
    def mlp_uniql_class_name(self):
        return "NemotronHUniQLMLP"

    @property
    def mlp_w4a16_uniql_class_name(self):
        return "W4A16NemotronHUniQLMLP"

    @property
    def get_mamba_full_intermediate_size(self):
        return self.config.mamba_num_heads * self.config.mamba_head_dim
    
    @property
    def get_mamba_full_ssm_state_size(self):
        return self.config.ssm_state_size

    def compress_mlp(self, layer, down_x, ratio):
        reduced_intermediate_size = 128 * int(round(ratio * layer.mixer.intermediate_size / 128.))
        up_proj, down_proj, gate_proj = mlp_compression(layer.mixer.up_proj, layer.mixer.down_proj, None, down_x,
                                                        reduced_intermediate_size, ridge_lambda=1)
        compressed_mlp = NemotronHUniQLMLP(layer.mixer.config, layer.mixer.layer_idx, flex_layer_ratio={"mlp_ratio": ratio})
        compressed_mlp.up_proj = up_proj
        compressed_mlp.down_proj = down_proj
        layer.mixer = compressed_mlp
        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_intermediate_size" not in self.config.compress_config:
            self.config.compress_config["reduced_intermediate_size"] = {}
        self.config.compress_config["reduced_intermediate_size"][layer.mixer.layer_idx] = reduced_intermediate_size

    def replace_simple_mlp(self, layer, use_had_transform=False):
        layer.mixer = NemotronHMLPSimple(layer.mixer, use_had_transform=use_had_transform)
        
    def replace_w4a16_mlp(self, layer):
        if layer.mixer.__class__.__name__ == "NemotronHMLP":
            layer.mixer = W4A16NemotronHMLP.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHMLPSimple":
            layer.mixer = W4A16NemotronHMLP.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHUniQLMLP":
            layer.mixer = W4A16NemotronHUniQLMLP.from_fp16(layer.mixer)
        else:
            raise ValueError(f"Unsupported MLP class: {layer.mixer.__class__.__name__}")

    @property
    def layer_mlp_ops(self):
        return ["mixer.up_proj", "mixer.down_proj"]

    @property
    def layer_mlp_svd_ops(self):
        return ["mixer.up_proj", "mixer.down_proj"]

    @property
    def mamba_class_name(self):
        return "NemotronHMamba2Mixer"

    @property
    def mamba_simple_class_name(self):
        return "NemotronHMamba2MixerSimple"
    
    @property
    def mamba_uniql_class_name(self):
        return "NemotronHUniQLMamba2Mixer"

    @property
    def mamba_w4a16_uniql_class_name(self):
        return "W4A16NemotronHUniQLMamba2Mixer"

    @torch.no_grad()
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
                                                  eps=layer.mixer.norm.variance_epsilon,
                                                  group_size=reduced_intermediate_size // layer.mixer.n_groups)
        x_proj, out_proj, z_proj, compressed_gated_norm, compressed_conv1d = ssd_state_aware_xo_compression(
                       layer.mixer.x_proj, layer.mixer.out_proj, layer.mixer.z_proj,
                       hidden_states, layer.mixer.B_proj, layer.mixer.dt_proj,
                       reduced_head_dim, layer.mixer.intermediate_size,
                       layer.mixer.n_groups,layer.mixer.num_heads, layer.mixer.head_dim,
                       layer.mixer.norm, compressed_gated_norm, layer.mixer.conv1d, compressed_conv1d,
                       layer.mixer.A_log, layer.mixer.chunk_size, layer.mixer.dt_bias,
                       dt_softplus=True, dt_limit=(0.0, float("inf")), seq_idx=None, ridge_lambda=1.0)

        compressed_mamba = NemotronHUniQLMamba2Mixer(layer.mixer.config, layer.mixer.layer_idx, flex_layer_ratio={"mamba_ratio": ratio})
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
        layer.mixer = NemotronHMamba2MixerSimple(layer.mixer, use_had_transform=use_had_transform)

    def replace_w4a16_mamba(self, layer):
        if layer.mixer.__class__.__name__ == "NemotronHMamba2Mixer":
            layer.mixer = W4A16NemotronHMamba2Mixer.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHUniQLMamba2Mixer":
            layer.mixer = W4A16NemotronHUniQLMamba2Mixer.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHMamba2MixerSimple":
            layer.mixer.fuse_in_proj()
            # layer.mixer = W4A16NemotronHMamba2Mixer.from_fp16(layer.mixer)
            if hasattr(layer.mixer.config, "compress_config"):
                # NemotronHUniQLMamba2Mixer -> NemotronHMamba2MixerSimple use this
                layer.mixer = W4A16NemotronHUniQLMamba2Mixer.from_fp16(layer.mixer)
            else:
                # NemotronHMamba2Mixer -> NemotronHMamba2MixerSimple use this
                layer.mixer = W4A16NemotronHMamba2Mixer.from_fp16(layer.mixer)
        else:
            raise ValueError(f"Unsupported Mamba class: {layer.mixer.__class__.__name__}")

    @property
    def layer_mamba_ops(self):
        return ["mixer.out_proj", "mixer.z_proj",  "mixer.x_proj",  "mixer.B_proj",  "mixer.C_proj",  "mixer.dt_proj"]

    @property
    def layer_mamba_svd_ops(self):
        return ["mixer.out_proj", "mixer.z_proj",  "mixer.x_proj"]

    @property
    def attn_class_name(self):
        return "NemotronHAttention"

    @property
    def attn_simple_class_name(self):
        return "NemotronHAttentionSimple"
    
    @property
    def attn_uniql_class_name(self):
        return "NemotronHUniQLAttention"

    @property
    def attn_w4a16_uniql_class_name(self):
        return "W4A16NemotronHUniQLAttention"

    def compress_attn(self, layer, qkv_x, ratio, position_embedding):
        head_dim = layer.mixer.head_dim
        reduced_head_dim = 16 * int(round(head_dim * ratio / 16.))
        q_proj, k_proj, indices = attn_qk_compression_gqa(layer.mixer.q_proj, layer.mixer.k_proj, qkv_x, reduced_head_dim,
                                                          layer.mixer.config.num_attention_heads, layer.mixer.config.num_key_value_heads,
                                                          position_embeddings=position_embedding, rotary_fn=None, rotate_half=True)
        v_proj, o_proj = attn_vo_compression_gqa(layer.mixer.v_proj, layer.mixer.o_proj, qkv_x, reduced_head_dim,
                                                 layer.mixer.config.num_attention_heads, layer.mixer.config.num_key_value_heads)
        compressed_self_attn = NemotronHUniQLAttention(layer.mixer.config, layer.mixer.layer_idx, flex_layer_ratio={"attn_ratio": ratio})
        compressed_self_attn.q_proj = q_proj
        compressed_self_attn.k_proj = k_proj
        compressed_self_attn.v_proj = v_proj
        compressed_self_attn.o_proj = o_proj
        compressed_self_attn.indices = indices.to(torch.uint8)
        layer.mixer = compressed_self_attn
        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_head_dim" not in self.config.compress_config:
            self.config.compress_config["reduced_head_dim"] = {}
        self.config.compress_config["reduced_head_dim"][layer.mixer.layer_idx] = reduced_head_dim

    def replace_simple_attn(self, layer, use_had_transform=False):
        layer.mixer = NemotronHAttentionSimple(layer.mixer, use_had_transform=use_had_transform)

    def replace_w4a16_attn(self, layer):
        if layer.mixer.__class__.__name__ == "NemotronHAttention":
            layer.mixer = W4A16NemotronHAttention.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHUniQLAttention":
            layer.mixer = W4A16NemotronHUniQLAttention.from_fp16(layer.mixer)
        elif layer.mixer.__class__.__name__ == "NemotronHAttentionSimple":
            # layer.mixer = W4A16NemotronHAttention.from_fp16(layer.mixer)
            # if hasattr(layer.mixer, "indices") and layer.mixer.indices is not None: # NemotronH does not have embedding indices
            if hasattr(layer.mixer.config, "compress_config"):
                # NemotronHUniQLMixer -> NemotronHAttentionSimple
                layer.mixer = W4A16NemotronHUniQLAttention.from_fp16(layer.mixer)
            else:
                # NemotronHAttention -> NemotronHAttentionSimple
                layer.mixer = W4A16NemotronHAttention.from_fp16(layer.mixer)
        else:
            raise ValueError(f"Unsupported Attention class: {layer.mixer.__class__.__name__}")
        
    @property
    def layer_attn_ops(self):
        return ["mixer.o_proj", "mixer.q_proj", "mixer.k_proj", "mixer.v_proj"]

    @property
    def layer_attn_svd_ops(self):
        return ["mixer.o_proj", "mixer.q_proj"]
    

