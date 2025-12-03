import os
import re
import time
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    BitsAndBytesConfig
)

### Llama
from modeling.llama.modeling_llama import LlamaForCausalLM
from modeling.llama.modeling_qllama import W4A16LlamaForCausalLM

### Bamba
from modeling.bamba.modeling_bamba import BambaForCausalLM
from modeling.bamba.modeling_qbamba import W4A16BambaForCausalLM

## Nemotron-H
from modeling.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from modeling.nemotron_h.modeling_qnemotron_h import W4A16NemotronHForCausalLM

### Mamba2
from transformers import T5Tokenizer
from utils.megatron_utils import _GPTSentencePieceTokenizer
from modeling.mamba.modeling_mamba2 import Mamba2ForCausalLM
from modeling.mamba.modeling_qmamba2 import W4A16Mamba2ForCausalLM

### Qwen2
from modeling.qwen.modeling_qwen2 import Qwen2ForCausalLM
from modeling.qwen.modeling_qqwen2 import W4A16Qwen2ForCausalLM
from modeling.qwen.tokenization_qwen2 import Qwen2Tokenizer

from utils.model_utils import contains_wxa16
from utils.model_utils import get_wxa16_number
from utils.logger_utils import set_logger

logger = logging.getLogger(os.path.basename(__file__))
set_logger(logger, logging.INFO)

def build_model_and_tokenizer(model_repo, pretrained_dir=None, layer_ratio_config=None, dtype=torch.float16):
    start = time.time()
    logger.info(f"Loading model and tokenizer from {model_repo}")
    # create model and tokenizer on GPU
    load_model_path = model_repo # huggingface path
    if "ut-enyac" in model_repo:
        assert pretrained_dir is not None, "pretrained_dir must be provided for ut-enyac pretrained models"
        load_model_path = os.path.join(pretrained_dir, model_repo) # load from local directory
        assert os.path.isdir(load_model_path), f"Model directory {load_model_path} does not exist."
    
    if contains_wxa16(load_model_path):
        w_bits = get_wxa16_number(load_model_path)
        if "llama" in load_model_path.lower():
            if "llama-7b-hf" in load_model_path.lower():
                # for jeffwan/llama-7b-hf, which is used in svd-llm, this does not work with Llama-3
                tokenizer = LlamaTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
                if w_bits == 4: 
                    model = W4A16LlamaForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
                else:
                    logger.error(f"We can only support w{w_bits}a16 models.")
                    exit(1)
        elif "bamba" in load_model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            if w_bits == 4: 
                model = W4A16BambaForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
            else:
                logger.error(f"We can only support w{w_bits}a16 models.")
                exit(1)
        elif "nemotron-h" in load_model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            if w_bits == 4:
                model = W4A16NemotronHForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
            else:
                logger.error(f"We can only support w{w_bits}a16 models.")
                exit(1)
        elif "mamba2" in load_model_path.lower():
            # tokenizer_ckpt = os.path.join(load_model_path, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            # tokenizer = _GPTSentencePieceTokenizer(os.path.abspath(tokenizer_ckpt)) # must use abspath to make padding work...not sure why...
            # Use T5Tokenizer to make things easier for evaluation and LoRA finetuning...
            # tokenizer = T5Tokenizer.from_pretrained(os.path.abspath(tokenizer_ckpt), trust_remote_code=True)
            tokenizer = T5Tokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            if w_bits == 4:
                model = W4A16Mamba2ForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
            else:
                logger.error(f"We can only support w{w_bits}a16 models.")
                exit(1)
        elif "qwen2" in load_model_path.lower():
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
            if w_bits == 4:
                model = W4A16Qwen2ForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
            else:
                logger.error(f"We can only support w{w_bits}a16 models.")
                exit(1)
        else:
            raise ValueError(f"Unsupported model type: {load_model_path}")
        # due to our w4a16linear implementation, we can only use torch.float16
        if dtype != torch.float16:
            logger.warning(f"Converting model to torch.float16, as w4a16 models only support torch.float16")
        model = model.to(torch.float16)
    else:
        if "llama" in load_model_path.lower():
            if "llama-7b-hf" in load_model_path.lower():
                # for jeffwan/llama-7b-hf, which is used in svd-llm, this does not work with Llama-3
                tokenizer = LlamaTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            model = LlamaForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
        elif "qwen2" in load_model_path.lower():
            tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
            model = Qwen2ForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
        elif "bamba" in load_model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            model = BambaForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
        elif "nemotron-h" in load_model_path.lower():
            # Nemotron-H is more stable with torch.bfloat16
            tokenizer = AutoTokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            model = NemotronHForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
        elif "mamba2" in load_model_path.lower():
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            # tokenizer_ckpt = os.path.join(load_model_path, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            # tokenizer = _GPTSentencePieceTokenizer(os.path.abspath(tokenizer_ckpt)) # must use abspath to make padding work...not sure why...
            # NOTE(hychiang): Use T5Tokenizer for evaluating mamba2-8b to avoid the assertion
            # isinstance(tokenizer, transformers.PreTrainedTokenizer) at lm_eval/models/huggingface.py"
            # Use T5Tokenizer to make things easier for evaluation and LoRA finetuning...
            # tokenizer = T5Tokenizer.from_pretrained(os.path.abspath(tokenizer_ckpt), trust_remote_code=True)
            tokenizer = T5Tokenizer.from_pretrained(load_model_path, trust_remote_code=True)
            model = Mamba2ForCausalLM.from_pretrained(load_model_path, trust_remote_code=True, layer_ratio_config=layer_ratio_config)
        else:
            raise ValueError(f"Unsupported model type: {load_model_path}")
        # convert the model to the desired data type
        model = model.to(dtype)
    model = model.cuda()
    model.eval()
    elaspe_time = time.time() - start
    logger.info(f"Load model and tokenizer takes: {elaspe_time:.2f} s")
    return model, tokenizer
