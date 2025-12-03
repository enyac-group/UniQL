import logging
import torch
import torch.nn as nn
from transformers.cache_utils import StaticCache
from modeling.module_helpers import get_size
# qwen2
from modeling.qwen.qwen2_blocks import apply_rotary_pos_emb
from modeling.qwen.qwen2_simple_blocks import (
    Qwen2MLPSimple,
    Qwen2AttentionSimple,
)
from modeling.qwen.qwen2_uniql_blocks import (
    Qwen2UniQLMLP,
    Qwen2UniQLAttention
)
from modeling.qwen.qwen2_qblocks import (
    W4A16Qwen2MLP,
    W4A16Qwen2Attention,
)
from modeling.qwen.qwen2_uniql_qblocks import (
    W4A16Qwen2UniQLMLP,
    W4A16Qwen2UniQLAttention,
)

from compress.compress_helpers import mlp_compression
from compress.compress_helpers import attn_qk_compression, attn_vo_compression
from compress.compress_helpers import attn_qk_compression_gqa, attn_vo_compression_gqa

class Qwen2Helper():

    def __init__(self, model_config):
        self.config = model_config
        self.module_mapping = {
            "Qwen2Attention": "attn",
            "Qwen2UniQLAttention": "attn",
            "Qwen2AttentionSimple": "attn",
            "W4A16Qwen2Attention": "attn",
            "W4A16Qwen2UniQLAttention": "attn",
            "Qwen2MLP": "mlp",
            "Qwen2UniQLMLP": "mlp",
            "Qwen2MLPSimple": "mlp",
            "W4A16Qwen2MLP": "mlp",
            "W4A16Qwen2UniQLMLP": "mlp",
            "Qwen2RMSNorm": "norm",
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
        module_size["norm"] += get_size(model.model.norm)
        module_size["embedding"] = get_size(model.model.embed_tokens)
        module_size["output"] = 0 if model.lm_head.weight is model.model.embed_tokens.weight else get_size(model.lm_head)
        module_size["others"] = get_size(model.model.rotary_emb)
        return module_size
    
    def get_cache_size(self, batch_size, prompt_len):
        cache = StaticCache(config=self.config, max_batch_size=batch_size, max_cache_len=prompt_len, device=None)
        kv_cache_size = 0
        for k_cache in cache.key_cache:
            kv_cache_size += k_cache.nelement() * k_cache.element_size()
        for v_cache in cache.value_cache:
            kv_cache_size += v_cache.nelement() * v_cache.element_size()
        return 0, 0, kv_cache_size

    def get_embeddings(self, model):
        return model.model.embed_tokens

    def set_embeddings(self, model, new_embedding):
        model.model.embed_tokens = new_embedding

    def get_layers(self, model):
        return model.model.layers

    @property
    def layer_step(self):
        # a layer means a mixer block plus a mlp block in Llama
        return 1
    
    def get_input_layernorm(self, layer):
        return layer.input_layernorm

    def get_mlp_norm(self, layer):
        return layer.post_attention_layernorm
    
    def get_mlp(self, layer):
        return layer.mlp

    def get_mlp_key(self, layer_idx):
        return f"model.layers.{layer_idx}.mlp"

    def get_mamba_norm(self, layer):
        raise NotImplementedError("Qwen2 does not have Mamba layers.")
    
    def get_mamba(self, layer):
        raise NotImplementedError("Qwen2 does not have Mamba layers.")

    def get_mamba_key(self, layer_idx):
        raise NotImplementedError("Qwen2 does not have Mamba layers.")

    def get_attn_norm(self, layer):
        return layer.input_layernorm
    
    def get_attn(self, layer):
        return layer.self_attn

    def get_attn_key(self, layer_idx):
        return f"model.layers.{layer_idx}.self_attn"

    def get_final_layernorm(self, model):
        return model.model.norm
    
    def get_lm_head(self, model):
        return model.lm_head

    def set_lm_head(self, model, new_head):
        model.lm_head = new_head

    @property
    def mlp_class_name(self):
        return "Qwen2MLP"

    @property
    def mlp_simple_class_name(self):
        return "Qwen2MLPSimple"
    
    @property
    def mlp_uniql_class_name(self):
        return "Qwen2UniQLMLP"

    @property
    def mlp_w4a16_uniql_class_name(self):
        return "W4A16Qwen2UniQLMLP"

    def compress_mlp(self, layer, down_x, ratio):
        reduced_intermediate_size = 128 * int(round(ratio * layer.mlp.intermediate_size / 128.))
        up_proj, down_proj, gate_proj = mlp_compression(layer.mlp.up_proj, layer.mlp.down_proj, layer.mlp.gate_proj, down_x,
                                                        reduced_intermediate_size, ridge_lambda=1)
        compressed_mlp = Qwen2UniQLMLP(layer.mlp.config, layer.mlp.layer_idx, flex_layer_ratio={"mlp_ratio": ratio})
        compressed_mlp.up_proj = up_proj
        compressed_mlp.down_proj = down_proj
        compressed_mlp.gate_proj = gate_proj
        layer.mlp = compressed_mlp
        if not hasattr(self.config, "compress_config"):
            self.config.compress_config = {}
        if "reduced_intermediate_size" not in self.config.compress_config:
            self.config.compress_config["reduced_intermediate_size"] = {}
        self.config.compress_config["reduced_intermediate_size"][layer.mlp.layer_idx] = reduced_intermediate_size

    def replace_simple_mlp(self, layer, use_had_transform=False):
        layer.mlp = Qwen2MLPSimple(layer.mlp, use_had_transform=use_had_transform)
    
    def replace_w4a16_mlp(self, layer):
        if layer.mlp.__class__.__name__ == "Qwen2MLP":
            layer.mlp = W4A16Qwen2MLP.from_fp16(layer.mlp)
        elif layer.mlp.__class__.__name__ == "Qwen2MLPSimple":
            layer.mlp = W4A16Qwen2MLP.from_fp16(layer.mlp)
        elif layer.mlp.__class__.__name__ == "Qwen2UniQLMLP":
            layer.mlp = W4A16Qwen2UniQLMLP.from_fp16(layer.mlp)
        else:
            raise ValueError(f"Unsupported MLP class: {layer.mlp.__class__.__name__}")

    @property
    def layer_mlp_ops(self):
        return ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]

    @property
    def mamba_class_name(self):
        return None

    @property
    def mamba_simple_class_name(self):
        return None

    @property
    def mamba_uniql_class_name(self):
        return None

    @property
    def mamba_w4a16_uniql_class_name(self):
        return None
    
    def replace_simple_mamba(self, layer):
        raise NotImplementedError("Qwen2 does not have Mamba layers.")

    def replace_w4a16_mamba(self, layer):
        raise NotImplementedError("Qwen2 does not have Mamba layers.")
    
    @property
    def layer_mamba_ops(self):
        return []

    @property
    def attn_class_name(self):
        return "Qwen2Attention"

    @property
    def attn_simple_class_name(self):
        return "Qwen2AttentionSimple"
    
    @property
    def attn_uniql_class_name(self):
        return "Qwen2UniQLAttention"

    @property
    def attn_w4a16_uniql_class_name(self):
        return "W4A16Qwen2UniQLAttention"

    def compress_attn(self, layer, qkv_x, ratio, position_embedding):

        if self.config.num_attention_heads != self.config.num_key_value_heads:
            # GQA is used for models with different number of attention heads and key value heads
            attn_qk_compression_fn = attn_qk_compression_gqa
            attn_vo_compression_fn = attn_vo_compression_gqa
        else:
            attn_qk_compression_fn = attn_qk_compression
            attn_vo_compression_fn = attn_vo_compression

        head_dim = layer.self_attn.head_dim
        reduced_head_dim = 16 * int(round(head_dim * ratio / 16.))
        q_proj, k_proj, indices = attn_qk_compression_fn(layer.self_attn.q_proj, layer.self_attn.k_proj, qkv_x, reduced_head_dim,
                                                         layer.self_attn.config.num_attention_heads, layer.self_attn.config.num_key_value_heads,
                                                         position_embeddings=position_embedding, rotary_fn=apply_rotary_pos_emb, rotate_half=False)
        v_proj, o_proj = attn_vo_compression_fn(layer.self_attn.v_proj, layer.self_attn.o_proj, qkv_x, reduced_head_dim,
                                                layer.self_attn.config.num_attention_heads, layer.self_attn.config.num_key_value_heads)
        compressed_self_attn = Qwen2UniQLAttention(layer.self_attn.config, layer.self_attn.layer_idx, flex_layer_ratio={"attn_ratio": ratio})
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
        layer.self_attn = Qwen2AttentionSimple(layer.self_attn, use_had_transform=use_had_transform)

    def replace_w4a16_attn(self, layer):
        if layer.self_attn.__class__.__name__ == "Qwen2Attention":
            layer.self_attn = W4A16Qwen2Attention.from_fp16(layer.self_attn)
        elif layer.self_attn.__class__.__name__ == "Qwen2UniQLAttention":
            layer.self_attn = W4A16Qwen2UniQLAttention.from_fp16(layer.self_attn)
        elif layer.self_attn.__class__.__name__ == "Qwen2AttentionSimple":
            # layer.self_attn = W4A16Qwen2Attention.from_fp16(layer.self_attn)
            if hasattr(layer.self_attn, "indices") and layer.self_attn.indices is not None:
                # Qwen2UniQLMixer -> Qwen2MixerSimple
                layer.self_attn = W4A16Qwen2UniQLAttention.from_fp16(layer.self_attn)
            else:
                # Qwen2Mixer -> Qwen2MixerSimple
                layer.self_attn = W4A16Qwen2Attention.from_fp16(layer.self_attn)
        else:
            raise ValueError(f"Unsupported Attention class: {layer.self_attn.__class__.__name__}")

    @property
    def layer_attn_ops(self):
        return ["self_attn.o_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
