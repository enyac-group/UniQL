import re
import os
import logging
from modeling.model_helper_registry import ModelHelperRegistry
from utils.logger_utils import set_logger 

logger = logging.getLogger(os.path.basename(__file__))
set_logger(logger, logging.INFO)

def contains_wxa16(name):
    return bool(re.search(r'w\d+a16', name))

def get_wxa16_number(name):
    """Extract the number x from wxa16 pattern (e.g., 'w4a16' -> 4)"""
    match = re.search(r'w(\d+)a16', name)
    if match:
        return int(match.group(1))
    return None

def extract_layer_number(module_name):
    """Extract layer number from module name.
    
    Args:
        module_name (str): Module name like 'base_model.model.model.layers.31.mlp.down_proj.lora_B.default.weight'
    
    Returns:
        int or None: Layer number if found, None otherwise
        
    Examples:
        >>> extract_layer_number("base_model.model.model.layers.31.mlp.down_proj.lora_B.default.weight")
        31
        >>> extract_layer_number("model.layers.0.self_attn.q_proj.weight")
        0
    """
    match = re.search(r'layers\.(\d+)', module_name)
    if match:
        return int(match.group(1))
    return None

def extract_base_model_name(model_name):
    """Extract the base model name without compression/quantization suffixes.
    
    Args:
        model_name (str): Model name that may contain suffixes like -uniql-0.75, -w3a16, etc.
    
    Returns:
        str: Base model name without suffixes
        
    Examples:
        >>> extract_base_model_name("Bamba-9B-v2-uniql-0.75")
        "Bamba-9B-v2"
        >>> extract_base_model_name("Bamba-9B-v2-w3a16")
        "Bamba-9B-v2"
    """
    # Remove common compression/quantization suffixes
    suffixes_to_remove = [
        r'-uniql-\d+\.\d+',    # -uniql-0.75
        r'-w\d+a\d+',          # -w3a16, -w4a16, etc.
        r'-quantized',         # -quantized
        r'-compressed',        # -compressed
        r'-uniqlow',           # -uniqlow
        r'-gptq',              # -gptq
        r'-wikitext2',         # -wikitext2
    ]
    
    cleaned_name = model_name
    for suffix_pattern in suffixes_to_remove:
        cleaned_name = re.sub(suffix_pattern, '', cleaned_name)
    
    return cleaned_name

def get_saved_model_name(model_repo, postfixs=None):
    model_name = model_repo.split('/')[-1]
    if model_name.startswith('Meta'): # Meta-Llama-3-8B
        model_name = "-".join(model_name.split('-')[1:]) #  Meta-Llama-3-8B -> Llama-3-8B
    for postfix in postfixs or []:
        model_name = "-".join([model_name, postfix]) # Nemotron-H-8B-Base-8K -> Nemotron-H-8B-Base-8K-low-rank
    return model_name


def model_name_and_type(model_repo):
    """ Extracts the model name and type from the model repository string.
    Args:
        model_repo (str): The model repository string, e.g., "meta-llama/Llama-3.1-8B".
    Returns:
        tuple: A tuple containing the model name and model type.
                For example, ("Llama-3.1-8B", "llama").
    """
    if not isinstance(model_repo, str):
        raise ValueError("model_repo must be a string")

    model_name = model_repo.lower().split('/')[-1]
    # we clean the model name for convinience
    if model_name.startswith('meta'): # Meta-Llama-3-8B
        model_name = "-".join(model_name.split('-')[1:]) #  Meta-Llama-3-8B -> Llama-3-8B
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    return model_name, model_type


def profile_size(model, model_type, batch_size=1, prompt_len=1024, use_GiB=False):

    logger.info(">>> Profiling model size")
    logger.info("Start profiling...")
    # get model total size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # create model abs
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    # model conv/ssm/kv caches
    GB = 1000**3 # This is the SI (base-10) definition used by most storage manufacturers.
    MB = 1000**2
    if use_GiB:
        GB = 1024**3 # Binary (used by Linux/OS), aka GiB
        MB = 1024**2 # Binary (used by Linux/OS), aka MiB
    conv_state_size, ssm_state_size, kv_cache_size = model_helper.get_cache_size(batch_size, prompt_len)
    cache_gb = (conv_state_size + ssm_state_size + kv_cache_size) / GB
    logger.info(f'cache size: {cache_gb:.3f} GB (batch size {batch_size}, prompt length {prompt_len}), detailed breakdown:')
    
    # Only log conv/ssm states if they exist (for Mamba-based models)
    if conv_state_size > 0:
        logger.info(f"-- conv state: {conv_state_size / GB:.3f} GB")
    if ssm_state_size > 0:
        logger.info(f"-- ssm state: {ssm_state_size / GB:.3f} GB")
    logger.info(f"-- kv cache: {kv_cache_size / GB:.3f} GB")
    
    # model total size and detailed layer type breakdown
    model_size = (param_size + buffer_size)
    layer_size_dict = model_helper.get_layer_size(model)
    logger.info(f'model size: {model_size / GB:.3f} GB, detailed breakdown:')
    layer_size_sum = 0
    for k, v in layer_size_dict.items():
        if v < MB:
            logger.info(f"-- {k}: {v / MB:.3f} MB")
        else:
            logger.info(f"-- {k}: {v / GB:.3f} GB")
        layer_size_sum += v
    assert layer_size_sum == model_size, f"Model size breakdown does not match the total model size: {layer_size_sum} != {model_size}"