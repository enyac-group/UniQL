import os
import json
import shutil
import logging
import argparse

import torch
from transformers import T5Tokenizer
from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

from utils.logger_utils import set_logger 
logger = logging.getLogger(os.path.basename(__file__))
set_logger(logger, logging.INFO)


def find_shared_params(model):
    if isinstance(model, dict):
        iterator = model.items()
    else:
        iterator = model.named_parameters()
    seen = {}
    shared = []
    for name, param in iterator:
        ptr = param.data_ptr()
        if ptr in seen:
            shared.append((name, seen[ptr]))
        else:
            seen[ptr] = name
    return shared



def main(args):
    # Step 1: Load and convert state_dict
    source_state_dict_path = args.source_state_dict_path
    
    # Load source state_dict
    source_state_dict = torch.load(source_state_dict_path, map_location='cpu', weights_only=False)['model']
    target_state_dict = {}
    
    # Define specific mappings for layers
    specific_mappings = {
        "embedding.word_embeddings.weight": "backbone.embedding.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        "output_layer.weight": "lm_head.weight"
    }
    
    logger.info("Start converting state_dict...")
    # Convert layers dynamically for layers 0-55
    for layer_idx in range(56):
        source_prefix = f'decoder.layers.{layer_idx}'
        target_prefix = f'backbone.layers.{layer_idx}'
        
        for source_key, value in source_state_dict.items():
            if source_key.startswith(source_prefix):
                target_key = source_key.replace(source_prefix, target_prefix)
                target_state_dict[target_key] = value
    # Apply specific mappings
    for source_key, target_key in specific_mappings.items():
        if source_key in source_state_dict:
            target_state_dict[target_key] = source_state_dict[source_key]

    dtype = target_state_dict["backbone.embedding.weight"].dtype
    logger.info(f"model dtype: {dtype}")

    shared_weights = find_shared_params(target_state_dict)
    for name, param_ptr in shared_weights:
        logger.warning(f"Shared weight: {name}")

    # Step 2: Load model AntonV/mamba2-2.7b-hf config and modify it for mamba2-8b-3t-4k
    logger.info("Create config file for mamba2-8b-hf...")
    resolved_archive_file = cached_file("AntonV/mamba2-2.7b-hf", CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    config_data = json.load(open(resolved_archive_file))
    config_data['chunk_size'] = 128
    config_data['n_groups'] = 8
    config_data['num_heads'] = 128
    config_data['hidden_size'] = 4096
    config_data['num_hidden_layers'] = 56
    config_data['vocab_size'] = 256000
    config_data['tie_word_embeddings'] = False # source_state_dict["embedding.word_embeddings.weight"] is source_state_dict["output_layer.weight"] == False
    config = Mamba2Config(**config_data)
    logger.info(config)
    
    # Initialize mamba2-8b-hf
    logger.info("Create mamba2-8b-hf from config...")
    # huggingface can only create fp32 model in this way, so it is slow and we have to convert it to the dtype after it is created
    model = Mamba2ForCausalLM(config)
    model = model.to(dtype)
    shared_weights = find_shared_params(model)
    for name, param_ptr in shared_weights:
        logger.warning(f"Shared weight: {name}")
    
    # Load the converted mamba2-8b-hf state_dict
    logger.info("Try to load the converted state_dict...")
    incompatible_key = model.load_state_dict(target_state_dict, strict=True)
    logger.warning(incompatible_key)
    logger.info("Model loaded successfully.")

    # Save mamba2-8b-hf for future use
    logger.info(f"Save the converted mamba2-8b-hf to {args.model_save_path}")
    del source_state_dict, target_state_dict
    shared_weights = find_shared_params(model)
    for name, param_ptr in shared_weights:
        logger.warning(f"Shared weight: {name}")
    # Not sure why we have this error if not use safe_serialization=False, but it seems working
    # RuntimeError: The weights trying to be saved contained shared tensors 
    # [{'backbone.embeddings.weight', 'lm_head.weight'}] that are mismatching the transformers base configuration.
    model_save_path = args.model_save_path
    model.save_pretrained(model_save_path, max_shard_size="4GB", safe_serialization=False)
    
    # try to load vocab_file with T5Tokenizer, and save it
    logger.info("Try to load vocab_file with T5Tokenizer, and save it...")
    vocab_file = args.vocab_file_path
    tokenizer = T5Tokenizer(vocab_file=vocab_file)
    tokenizer.save_pretrained(model_save_path)
    # Copy the original vocal_file for back up
    vocab_filename = os.path.basename(vocab_file)
    shutil.copyfile(vocab_file, os.path.join(args.model_save_path, vocab_filename))
    logger.info("Conversion, loading, and saving complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert and load state_dict for MambaLMHeadModel")
    parser.add_argument('source_state_dict_path', type=str, help="Path to the source state_dict file")
    parser.add_argument("vocab_file_path", type=str, help="Path to the vocab file")
    parser.add_argument('--model_save_path', type=str, default='./mamba2-8b-3t-4k_converted',
                        help="Path to save the converted model and tokenizer (default: ./mamba2-8b-3t-4k_converted)")    
    args = parser.parse_args()
    main(args)