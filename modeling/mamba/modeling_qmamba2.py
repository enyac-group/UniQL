# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
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
"""PyTorch MAMBA2 model."""

import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    logging,
)

from .configuration_mamba2 import Mamba2Config
from .modeling_mamba2 import Mamba2Output
from .modeling_mamba2 import Mamba2CausalLMOutput
from .mamba2_blocks import Mamba2Cache
from .mamba2_blocks import Mamba2RMSNorm
from .mamba2_qblocks import W4A16Mamba2Block
from .mamba2_uniql_qblocks import W4A16Mamba2UniQLBlock

from quantize.quant_linear_layers import W4A16B16O16Linear
from quantize.quant_embedding_layers import W4O16Embedding
from modeling.modeling_helpers import slice_quantized_weights
from modeling.modeling_helpers import load_config_hf, load_state_dict_hf
from utils.model_utils import model_name_and_type
from modeling.model_helper_registry import ModelHelperRegistry

import gc

logger = logging.get_logger(__name__)


@auto_docstring
class QMamba2PreTrainedModel(PreTrainedModel):
    config: Mamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["Mamba2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True


@dataclass
@auto_docstring(
    custom_intro="""
    Class for the MAMBA2 model outputs.
    """
)
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->Mamba2
class W4A16Mamba2Output(ModelOutput):
    r"""
    cache_params (`Mamba2Cache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for causal language model (or autoregressive) outputs.
    """
)
# Copied from transformers.models.mamba.modeling_mamba.MambaCausalLMOutput with Mamba->Mamba2
class W4A16Mamba2CausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`Mamba2Cache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[Mamba2Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class W4A16Mamba2Model(QMamba2PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.embeddings = W4O16Embedding(config.vocab_size, config.hidden_size)
        block = W4A16Mamba2Block
        if hasattr(config, "compress_config"):
            layer_ratio_config = kwargs.pop("layer_ratio_config", None)
            block = partial(W4A16Mamba2UniQLBlock, flex_layer_ratio=layer_ratio_config)
        self.layers = nn.ModuleList([block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Mamba2Output]:
        r"""
        cache_params (`Mamba2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = Mamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                # cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
                cache_position = None
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The MAMBA2 Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    """
)
class W4A16Mamba2ForCausalLM(QMamba2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.backbone = W4A16Mamba2Model(config, **kwargs)
        self.lm_head = W4A16B16O16Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[Mamba2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        model_inputs = {"input_ids": input_ids.contiguous()}
        if use_cache and cache_params is None:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            cache_position = torch.arange(0, self.backbone.config.conv_kernel, device=input_ids.device)
            if inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}
                max_batch_size = inputs_embeds.size(0)
            else:
                max_batch_size = input_ids.size(0)
            cache_params = Mamba2Cache(self.backbone.config, max_batch_size, device=self.device, dtype=self.dtype)

        if use_cache and cache_position[0] > 0:
            model_inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1).contiguous()
            attention_mask = None

        if not use_cache and inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[Mamba2Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation and loss_function
    ) -> Union[tuple, Mamba2CausalLMOutput]:
        r"""
        cache_params (`Mamba2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba2_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = mamba2_outputs[0]

        # logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + mamba2_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Mamba2CausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba2_outputs.cache_params,
            hidden_states=mamba2_outputs.hidden_states,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        config_data = load_config_hf(pretrained_model_name, cache_dir=cache_dir)
        config = Mamba2Config(**config_data)
        model = cls(config, device="cpu", **kwargs) # we always load model on cpu first to save memory
        loaded_model = load_state_dict_hf(pretrained_model_name, device="cpu", cache_dir=cache_dir)
        if kwargs.get("layer_ratio_config", None) is not None:
            model_name, model_type = model_name_and_type(pretrained_model_name)
            model_helper = ModelHelperRegistry[model_type](model_config=model.config)
            loaded_model = slice_quantized_weights(model, model_helper, loaded_model)
            model.load_state_dict(loaded_model, strict=True)
        else:
            model.load_state_dict(loaded_model, strict=True)
        del loaded_model
        torch.cuda.empty_cache()
        gc.collect()
        return model.to(device)
