# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging
from .configuration_llama import LlamaConfig
from .llama_blocks import eager_attention_forward
from .llama_rope import apply_rotary_pos_emb, apply_rotary_pos_emb_indices

from quantize.quant_linear_layers import HadLinear

logger = logging.get_logger(__name__)


class LlamaMLPSimple(nn.Module):
    def __init__(self, mlp, use_had_transform=False):
        super().__init__()
        self.config = copy.deepcopy(mlp.config)
        self.layer_idx = mlp.layer_idx
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size

        if use_had_transform:
            self.gate_proj = HadLinear(mlp.gate_proj, input_transform=True, output_transform=False)
            self.up_proj = HadLinear(mlp.up_proj, input_transform=True, output_transform=False)
            self.down_proj = HadLinear(mlp.down_proj, input_transform=False, output_transform=True)
        else:
            self.gate_proj = mlp.gate_proj
            self.up_proj = mlp.up_proj
            self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(x)
        return down_proj



class LlamaAttentionSimple(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attn, use_had_transform = True):
        super().__init__()
        self.config = copy.deepcopy(attn.config)
        self.layer_idx = attn.layer_idx
        self.head_dim = attn.head_dim
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout
        self.is_causal = True

        if hasattr(self.config, "compress_config"):
            self.register_buffer("indices", attn.indices.clone(), persistent=True)
            self.apply_rotary_pos_emb_fn = apply_rotary_pos_emb_indices
        else:
            self.register_buffer("indices", None)
            self.apply_rotary_pos_emb_fn = apply_rotary_pos_emb # we abuse the deprecated position_ids argument to pass in indices=None

        # output proj
        if use_had_transform:
            self.q_proj = HadLinear(attn.q_proj, input_transform=True, output_transform=False)
            self.k_proj = HadLinear(attn.k_proj, input_transform=True, output_transform=False)
            self.v_proj = HadLinear(attn.v_proj, input_transform=True, output_transform=False)
            self.o_proj = HadLinear(attn.o_proj, input_transform=False, output_transform=True)
        else:
            self.q_proj = attn.q_proj
            self.k_proj = attn.k_proj
            self.v_proj = attn.v_proj
            self.o_proj = attn.o_proj


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb_fn(query_states, key_states, cos, sin, self.indices)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
