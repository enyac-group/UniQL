import copy
import torch
import torch.nn as nn

from modeling.module_helpers import get_size
from modeling.bamba.bamba_blocks import apply_rotary_pos_emb
from modeling.bamba.modeling_bamba import HybridMambaAttentionDynamicCache as BambaCache
from modeling.bamba.bamba_blocks import BambaRMSNormGated
from modeling.bamba.bamba_simple_blocks import (
    BambaMixerSimple,
    BambaAttentionSimple,
    BambaMLPSimple
)
from modeling.bamba.bamba_uniql_blocks import (
    BambaUniQLMLP,
    BambaUniQLMixer,
    BambaUniQLAttention,
)
from modeling.bamba.bamba_qblocks import (
    W4A16BambaMLP,
    W4A16BambaMixer,
    W4A16BambaAttention
)
from modeling.bamba.bamba_uniql_qblocks import (
    W4A16BambaUniQLMLP,
    W4A16BambaUniQLMixer,
    W4A16BambaUniQLAttention
)
from compress.compress_helpers import mlp_compression
from compress.compress_helpers import ssd_BC_compression, ssd_xo_compression
from compress.compress_helpers import attn_qk_compression_gqa, attn_vo_compression_gqa

class BambaHelper():

    def __init__(self, model_config):
        self.config = copy.deepcopy(model_config)
        self.module_mapping = {
            "BambaMixer": "ssm",
            "BambaMixerSimple": "ssm",
            "BambaUniQLMixer": "ssm",
            "W4A16BambaMixer": "ssm",
            "W4A16BambaUniQLMixer": "ssm",
            "BambaMLP": "mlp",
            "BambaMLPSimple": "mlp",
            "BambaUniQLMLP": "mlp",
            "W4A16BambaMLP": "mlp",
            "W4A16BambaUniQLMLP": "mlp",
            "BambaAttention": "attn",
            "BambaAttentionSimple": "attn",
            "BambaUniQLAttention": "attn",
            "W4A16BambaAttention": "attn",
            "W4A16BambaUniQLAttention": "attn",
            "BambaRMSNormGated": "norm",
            "BambaRMSNorm": "norm",
        }

    def get_layer_size(self, model):
        module_size = {module_type: 0 for class_name, module_type in self.module_mapping.items()}
        for layer_idx, layer in enumerate(model.model.layers):
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
        module_size["norm"] += get_size(model.model.final_layernorm)
        module_size["embedding"] = get_size(model.model.embed_tokens)
        module_size["output"] = 0 if model.lm_head.weight is model.model.embed_tokens.weight else get_size(model.lm_head)
        module_size["others"] = get_size(model.model.rotary_emb)
        return module_size
    
    def get_cache_size(self, batch_size, prompt_len):
        cache = BambaCache(self.config, batch_size, dtype=torch.float16, device=None)
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
        return model.model.embed_tokens

    def set_embeddings(self, model, new_embedding):
        model.model.embed_tokens = new_embedding

    def get_layers(self, model):
        return model.model.layers

    @property
    def layer_step(self):
        # a layer means a mixer block plus a mlp block in Bamba
        return 1

    def get_input_layernorm(self, layer):
        return layer.input_layernorm

    def get_mlp_norm(self, layer):
        return layer.pre_ff_layernorm
    
    def get_mlp(self, layer):
        return layer.feed_forward

    def get_mlp_key(self, layer_idx):
        return f"model.layers.{layer_idx}.feed_forward"

    def get_mamba_norm(self, layer):
        return layer.input_layernorm
    
    def get_mamba(self, layer):
        return layer.mamba

    def get_mamba_key(self, layer_idx):
        return f"model.layers.{layer_idx}.mamba"

    def get_attn_norm(self, layer):
        return layer.input_layernorm
    
    def get_attn(self, layer):
        return layer.self_attn

    def get_attn_key(self, layer_idx):
        return f"model.layers.{layer_idx}.self_attn"

    def get_final_layernorm(self, model):
        return model.model.final_layernorm
    
    def get_lm_head(self, model):
        return model.lm_head

    def set_lm_head(self, model, new_head):
        model.lm_head = new_head

    @property
    def mlp_class_name(self):
        return "BambaMLP"

    @property
    def mlp_simple_class_name(self):
        return "BambaMLPSimple"

    @property
    def mlp_uniql_class_name(self):
        return "BambaUniQLMLP"

    @property
    def mlp_w4a16_uniql_class_name(self):
        return "W4A16BambaUniQLMLP"

    @property
    def get_mamba_full_intermediate_size(self):
        return int(self.config.mamba_expand * self.config.hidden_size)

    @property
    def get_mamba_full_ssm_state_size(self):
        return self.config.mamba_d_state

    def compress_mlp(self, layer, down_x, ratio):
        reduced_intermediate_size = 128 * int(round(ratio * layer.feed_forward.intermediate_size / 128.))
        up_proj, down_proj, gate_proj = mlp_compression(layer.feed_forward.up_proj, layer.feed_forward.down_proj, layer.feed_forward.gate_proj, down_x,
                                                        reduced_intermediate_size, ridge_lambda=1)
        compressed_mlp = BambaUniQLMLP(layer.feed_forward.config, layer.feed_forward.layer_idx, flex_layer_ratio={"mlp_ratio": ratio})
        compressed_mlp.up_proj = up_proj
        compressed_mlp.down_proj = down_proj
        compressed_mlp.gate_proj = gate_proj
        layer.feed_forward = compressed_mlp
        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_intermediate_size" not in self.config.compress_config:
            self.config.compress_config["reduced_intermediate_size"] = {}
        self.config.compress_config["reduced_intermediate_size"][layer.feed_forward.layer_idx] = reduced_intermediate_size

    def replace_simple_mlp(self, layer, use_had_transform=False):
        layer.feed_forward = BambaMLPSimple(layer.feed_forward, use_had_transform=use_had_transform)
    
    def replace_w4a16_mlp(self, layer):
        if layer.feed_forward.__class__.__name__ == "BambaMLP":
            layer.feed_forward = W4A16BambaMLP.from_fp16(layer.feed_forward)
        elif layer.feed_forward.__class__.__name__ == "BambaMLPSimple":
            layer.feed_forward = W4A16BambaMLP.from_fp16(layer.feed_forward)
        elif layer.feed_forward.__class__.__name__ == "BambaUniQLMLP":
            layer.feed_forward = W4A16BambaUniQLMLP.from_fp16(layer.feed_forward)
        else:
            raise ValueError(f"Unsupported MLP class: {layer.mlp.__class__.__name__}")

    @property
    def layer_mlp_ops(self):
        return ["feed_forward.up_proj", "feed_forward.gate_proj", "feed_forward.down_proj"]

    @property
    def mamba_class_name(self):
        return "BambaMixer"

    @property
    def mamba_simple_class_name(self):
        return "BambaMixerSimple"

    @property
    def mamba_uniql_class_name(self):
        return "BambaUniQLMixer"

    @property
    def mamba_w4a16_uniql_class_name(self):
        return "W4A16BambaUniQLMixer"
    
    def compress_mamba(self, layer, hidden_states, y_normed, ratio):

        reduced_dstate = 16 * int(round(layer.mamba.ssm_state_size * ratio / 16.))
        B_proj, C_proj, compressed_conv1d = ssd_BC_compression(
            layer.mamba.B_proj, layer.mamba.C_proj, layer.mamba.dt_proj, hidden_states, reduced_dstate,
            layer.mamba.intermediate_size, layer.mamba.num_heads, layer.mamba.n_groups, layer.mamba.ssm_state_size, layer.mamba.conv1d)

        head_dim = layer.mamba.intermediate_size // layer.mamba.num_heads
        reduced_head_dim = 2 * int(round(head_dim * ratio / 2.))
        reduced_intermediate_size = reduced_head_dim * layer.mamba.num_heads
        compressed_gated_norm = BambaRMSNormGated(reduced_intermediate_size,
                                                  eps=layer.mamba.norm.variance_epsilon)
        x_proj, out_proj, z_proj, compressed_gated_norm, compressed_conv1d = ssd_xo_compression(
            layer.mamba.x_proj, layer.mamba.out_proj, layer.mamba.z_proj, y_normed, reduced_head_dim,
            layer.mamba.intermediate_size, layer.mamba.num_heads, layer.mamba.n_groups, layer.mamba.norm,
            compressed_gated_norm, compressed_conv1d, ridge_lambda=1)

        compressed_mamba = BambaUniQLMixer(layer.mamba.config, layer.mamba.layer_idx, flex_layer_ratio={"mamba_ratio": ratio})
        in_proj_weight = torch.cat([z_proj.weight.data, x_proj.weight.data,
                                    B_proj.weight.data, C_proj.weight.data,
                                    layer.mamba.dt_proj.weight.data], dim=0)
        compressed_mamba.in_proj.weight.data = in_proj_weight
        compressed_mamba.out_proj = out_proj
        compressed_mamba.norm = compressed_gated_norm
        compressed_mamba.conv1d = compressed_conv1d
        compressed_mamba.D = layer.mamba.D
        compressed_mamba.dt_bias = layer.mamba.dt_bias
        compressed_mamba.A_log = layer.mamba.A_log
        layer.mamba = compressed_mamba

        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_mamba_d_state" not in self.config.compress_config:
            self.config.compress_config["reduced_mamba_d_state"] = {}
        if "reduced_mamba_d_head" not in self.config.compress_config:
            self.config.compress_config["reduced_mamba_d_head"] = {}
        self.config.compress_config["reduced_mamba_d_state"][layer.mamba.layer_idx] = reduced_dstate
        self.config.compress_config["reduced_mamba_d_head"][layer.mamba.layer_idx] = reduced_head_dim


    def replace_simple_mamba(self, layer, use_had_transform=False):
        layer.mamba = BambaMixerSimple(layer.mamba, use_had_transform=use_had_transform)

    def replace_w4a16_mamba(self, layer):
        if layer.mamba.__class__.__name__ == "BambaMixer":
            layer.mamba = W4A16BambaMixer.from_fp16(layer.mamba)
        elif layer.mamba.__class__.__name__ == "BambaUniQLMixer":
            layer.mamba = W4A16BambaUniQLMixer.from_fp16(layer.mamba)
        elif layer.mamba.__class__.__name__ == "BambaMixerSimple":
            layer.mamba.fuse_in_proj()
            if hasattr(layer.mamba.config, "compress_config"):
                # BambaUniQLMixer -> BambaMixerSimple use this
                layer.mamba = W4A16BambaUniQLMixer.from_fp16(layer.mamba)
            else:
                # BambaMixer -> BambaMixerSimple use this
                layer.mamba = W4A16BambaMixer.from_fp16(layer.mamba)
        else:
            raise ValueError(f"Unsupported Mamba class: {layer.mamba.__class__.__name__}")

    @property
    def layer_mamba_ops(self):
        return ["mamba.out_proj", "mamba.z_proj", "mamba.x_proj", "mamba.B_proj", "mamba.C_proj", "mamba.dt_proj"]

    @property
    def attn_class_name(self):
        return "BambaAttention"

    @property
    def attn_simple_class_name(self):
        return "BambaAttentionSimple"

    @property
    def attn_uniql_class_name(self):
        return "BambaUniQLAttention"

    @property
    def attn_w4a16_uniql_class_name(self):
        return "W4A16BambaUniQLAttention"

    def compress_attn(self, layer, qkv_x, ratio, position_embedding):
        head_dim = layer.self_attn.head_dim
        reduced_head_dim = 16 * int(round(head_dim * ratio / 16.))
        q_proj, k_proj, indices = attn_qk_compression_gqa(layer.self_attn.q_proj, layer.self_attn.k_proj, qkv_x, reduced_head_dim,
                                                          layer.self_attn.config.num_attention_heads, layer.self_attn.config.num_key_value_heads,
                                                          position_embeddings=position_embedding, rotary_fn=apply_rotary_pos_emb, rotate_half=True)
        v_proj, o_proj = attn_vo_compression_gqa(layer.self_attn.v_proj, layer.self_attn.o_proj, qkv_x, reduced_head_dim,
                                                 layer.self_attn.config.num_attention_heads, layer.self_attn.config.num_key_value_heads)
        compressed_self_attn = BambaUniQLAttention(layer.self_attn.config, layer.self_attn.layer_idx, flex_layer_ratio={"attn_ratio": ratio})
        compressed_self_attn.q_proj = q_proj
        compressed_self_attn.k_proj = k_proj
        compressed_self_attn.v_proj = v_proj
        compressed_self_attn.o_proj = o_proj
        compressed_self_attn.indices = indices.to(torch.uint8)
        layer.self_attn = compressed_self_attn
        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_head_dim" not in self.config.compress_config:
            self.config.compress_config["reduced_head_dim"] = {}
        self.config.compress_config["reduced_head_dim"][layer.self_attn.layer_idx] = reduced_head_dim

    def replace_simple_attn(self, layer, use_had_transform=False):
        layer.self_attn = BambaAttentionSimple(layer.self_attn, use_had_transform=use_had_transform)

    def replace_w4a16_attn(self, layer):
        if layer.self_attn.__class__.__name__ == "BambaAttention":
            layer.self_attn = W4A16BambaAttention.from_fp16(layer.self_attn)
        elif layer.self_attn.__class__.__name__ == "BambaUniQLAttention":
            layer.self_attn = W4A16BambaUniQLAttention.from_fp16(layer.self_attn)
        elif layer.self_attn.__class__.__name__ == "BambaAttentionSimple":
            if hasattr(layer.self_attn, "indices") and layer.self_attn.indices is not None:
                # BambaUniQLMixer -> BambaMixerSimple
                layer.self_attn = W4A16BambaUniQLAttention.from_fp16(layer.self_attn)
            else:
                # BambaMixer -> BambaMixerSimple
                layer.self_attn = W4A16BambaAttention.from_fp16(layer.self_attn)
        else:
            raise ValueError(f"Unsupported Attention class: {layer.self_attn.__class__.__name__}")

    @property
    def layer_attn_ops(self):
        return ["self_attn.o_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
