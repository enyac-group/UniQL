from typing import Callable, Optional, Union

import torch
from torch import nn

import triton
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from .configuration_qwen2 import Qwen2Config
from .qwen2_blocks import Qwen2RMSNorm
from .qwen2_blocks import eager_attention_forward

from quantize.quant_linear_layers import W4A16B16O16Linear
if triton.__version__ >= "3.3.0":
    from modeling.rope_triton import apply_rotary_pos_emb_indices_triton as apply_rotary_pos_emb_indices
else:
    from .qwen2_rope import apply_rotary_pos_emb_indices


class W4A16Qwen2UniQLMLP(nn.Module):
    def __init__(self, config, layer_idx, flex_layer_ratio=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        # self.intermediate_size = config.intermediate_size
        if flex_layer_ratio is not None:
            # for tensorcore tiling to work, and to match quantization groups
            self.intermediate_size = 128 * int(round(self.config.intermediate_size * flex_layer_ratio["mlp_ratio"] / 128.))
        else:
            self.intermediate_size = config.compress_config["reduced_intermediate_size"][str(layer_idx)]
        self.gate_proj = W4A16B16O16Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = W4A16B16O16Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = W4A16B16O16Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    @classmethod
    def from_fp16(cls, originalLayer):
        # Create the quantized mixer using the same config
        qmlp = cls(originalLayer.config, originalLayer.layer_idx)
        # Replace linear layers with quantized versions
        qmlp.gate_proj = qmlp.gate_proj.from_fp16(originalLayer.gate_proj)
        qmlp.up_proj = qmlp.up_proj.from_fp16(originalLayer.up_proj)
        qmlp.down_proj = qmlp.down_proj.from_fp16(originalLayer.down_proj)
        return qmlp

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class W4A16Qwen2UniQLAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int, flex_layer_ratio=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        config_head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if flex_layer_ratio is not None:
            # for tensorcore tiling to work, 16 should match quantization groups because num_attention_heads is 32 in Llama2
            self.head_dim = 16 * int(round(config_head_dim * flex_layer_ratio["attn_ratio"] / 16.))
        else:
            self.head_dim = config.compress_config["reduced_head_dim"][str(layer_idx)]
        self.scaling = config_head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = W4A16B16O16Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = W4A16B16O16Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = W4A16B16O16Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False) # the v bias is fused with the o bias
        self.o_proj = W4A16B16O16Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True) # the fused v and o bias
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.register_buffer("indices", torch.empty((config.num_key_value_heads, self.head_dim), dtype=torch.uint8), persistent=True)

    @classmethod
    def from_fp16(cls, originalLayer):
        # take a LlamaAttention as an originalLayer 
        # Create the quantized mixer using the same config
        qattn = cls(originalLayer.config, originalLayer.layer_idx)
        qattn.q_proj = qattn.q_proj.from_fp16(originalLayer.q_proj)
        qattn.k_proj = qattn.k_proj.from_fp16(originalLayer.k_proj)
        qattn.v_proj = qattn.v_proj.from_fp16(originalLayer.v_proj)
        qattn.o_proj = qattn.o_proj.from_fp16(originalLayer.o_proj)
        qattn.indices = originalLayer.indices
        return qattn

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_indices(query_states, key_states, cos, sin, indices=self.indices)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class W4A16Qwen2UniQLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int, flex_layer_ratio=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = W4A16Qwen2UniQLAttention(config=config, layer_idx=layer_idx, flex_layer_ratio=flex_layer_ratio[layer_idx] if flex_layer_ratio is not None else None)
        self.mlp = W4A16Qwen2UniQLMLP(config, layer_idx=layer_idx, flex_layer_ratio=flex_layer_ratio[layer_idx] if flex_layer_ratio is not None else None)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states