import os
import re
import json
import random
import shutil
import logging
import argparse
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import transformers

from peft import (
    LoraConfig,
    get_peft_model,
)
from peft.peft_model import PeftModelForCausalLM

from modeling.model_helper_registry import ModelHelperRegistry
from modeling.build_models import build_model_and_tokenizer
from utils.model_utils import profile_size
from utils.model_utils import model_name_and_type
from utils.model_utils import get_saved_model_name
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_compress_options
from utils.args_utils import argparse_calibration_options
from utils.reproduce_utils import set_deterministic
from utils.dataset_utils import build_dataloader
from utils.logger_utils import set_logger

logger = logging.getLogger(os.path.basename(__file__))


# Define callback for random layer ratio updates during training
class RandomLayerRatioCallback(transformers.TrainerCallback):
    def __init__(self, model, model_helper, layer_ratio_config_dict):
        self.model = model
        self.model_helper = model_helper
        self.layer_ratio_config_dict = layer_ratio_config_dict
        self.layer_ratio_config_keys = list(self.layer_ratio_config_dict.keys())
        # we assign higher weights to lower ratios
        weights_exp = np.exp(-0.1 * np.arange(len(self.layer_ratio_config_keys)))
        self.weights_exp = weights_exp / weights_exp.sum()
        logger.info(f"layer_ratio_config_keys: {self.layer_ratio_config_keys}")
        logger.info(f"associated probabilities: {self.weights_exp}")

    def on_step_begin(self, args, state, control, **kwargs):
        """Update layer ratios randomly at the beginning of each training step"""
        # if state.global_step % 10 == 0:  # Log every 10 steps to avoid spam
        #     logger.debug(f"Updating layer ratios at step {state.global_step}")

        # Randomly select a new ratio configuration, lower ratios are more likely to be selected
        random_ratio = random.choices(self.layer_ratio_config_keys, weights=self.weights_exp, k=1)[0]
        layer_ratio_config = self.layer_ratio_config_dict[random_ratio]

        # Update all layers with the new ratio configuration
        layers = self.model_helper.get_layers(self.model)
        for layer_idx, layer in enumerate(layers):
            for name, module in layer.named_children():
                class_name = module.__class__.__name__
                if class_name in [self.model_helper.mlp_uniql_class_name]:
                    # logger.debug(f"layer_idx: {layer_idx}, class_name: {class_name}, mlp_ratio: {layer_ratio_config[layer_idx]['mlp_ratio']}")
                    module.set_ratio(layer_ratio_config[layer_idx]["mlp_ratio"])
                elif class_name in [self.model_helper.attn_uniql_class_name]:
                    # logger.debug(f"layer_idx: {layer_idx}, class_name: {class_name}, attn_ratio: {layer_ratio_config[layer_idx]['attn_ratio']}")
                    module.set_ratio(layer_ratio_config[layer_idx]["attn_ratio"])
                elif class_name in [self.model_helper.mamba_uniql_class_name]:
                    # logger.debug(f"layer_idx: {layer_idx}, class_name: {class_name}, mamba_ratio: {layer_ratio_config[layer_idx]['mamba_ratio']}")
                    module.set_ratio(layer_ratio_config[layer_idx]["mamba_ratio"])
                else:
                    pass

def get_layer_ratio_config(model_name, eps=0.1):
    compress_dir = os.path.dirname(os.path.abspath(__file__))
    # layer_config_dir = os.path.join(compress_dir, "outputs", extract_base_model_name(model_name))
    layer_config_dir = os.path.join(compress_dir, "outputs", model_name)
    logger.info(f"Loading layer ratio config from {layer_config_dir}")
    layer_config_files = glob(os.path.join(layer_config_dir, f"layerwise_eps-{eps}_ratio-*.json"))
    
    # Sort files by the "r" value in the filename
    def extract_ratio_from_filename(filename):
        # Extract the ratio value from filename like "layerwise_eps-0.1_ratio-0.5.json"
        match = re.search(r'ratio-([\d.]+)\.json$', filename)
        if match:
            return float(match.group(1))
        return 0.0  # Default value if pattern doesn't match
    
    layer_config_files.sort(key=extract_ratio_from_filename)

    layer_ratio_config = {}
    for config_file in layer_config_files:
        logger.info(f"Loading layer ratio config from {config_file}")
        ratio = extract_ratio_from_filename(config_file)
        layer_ratio_config[ratio] = json.load(open(config_file))
    return layer_ratio_config


def print_matched_layers(model, patterns):
    compiled_patterns = [re.compile(p) for p in patterns]

    logger.debug("Apollo optimized Layers:")
    for name, module in model.named_modules():
        for pattern in compiled_patterns:
            if pattern.match(name):
                logger.debug(f"optimizing layer {name}")
                # break  # stop after first match to avoid duplicates

def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug(f"training parameter {name}")

def main(args):

    logger.info(f"Fine-tune {args.model_repo}")
    logger.info(f"Use dataset {args.calib_data_repo}")
    logger.info(f"Number of samples: {args.calib_data_num}")
    logger.info(f"Sequence length: {args.calib_seqlen}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Total epochs: {args.epochs}")
    if not args.pretrained_dir:
        logger.warning("--pretrained_dir is not provided.")

    # build model and tokenizer, bfloat16 is more stable
    model, tokenizer = build_model_and_tokenizer(args.model_repo, pretrained_dir=args.pretrained_dir, dtype=torch.bfloat16)
    model_name, model_type = model_name_and_type(args.model_repo)
    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated for building model: {max_mem:.2f} GB")

    profile_size(model, model_type, batch_size=args.batch_size, prompt_len=1024)
    tokenizer.pad_token = tokenizer.eos_token # we have to pad here because we put all the data into a single batch

    # build dataloader for prompt-based finetuning
    train_loader, valid_loader = build_dataloader(
                                        args.calib_data_repo, tokenizer,
                                        batch_size=args.batch_size, num_sample=args.calib_data_num, max_length=args.calib_seqlen,
                                        columns=["input_ids", "attention_mask", "labels"], num_valid_split=2000,
                                        enable_instruct_prompting=True, instruct_output_length=args.calib_seqlen,
                                        torch_dataloader=False)

    # freeze all parameters first, we do not want to train embeddings, conv1d,norms, and lm_head
    for param in model.parameters():
        param.requires_grad = False
    
    # apply LoRA to the model
    if args.lora:
        logger.info(f"Apply LoRA to {model_name} ({model_type})")
        logger.info(f"lora_r: {args.lora_r}")
        logger.info(f"lora_alpha: {args.lora_alpha}")
        logger.info(f"lora_dropout: {args.lora_dropout}")
        logger.info(f"lora_target_modules: {args.lora_target_modules.split(",")}")
        # https://github.com/huggingface/peft/issues/2556, mamba out_proj is excluded by default.
        # We comment out the assertion in the peft library.
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        # for name, module in model.named_modules():
        #     if "lora" in name:
        #         logger.debug(f"{name}: {module}")
        # logger.info(f"trainable parameters with LoRA:")
        # model.print_trainable_parameters()
        if args.apollo_adamw:
            # we use Apollo AdamW optimizer to save memory
            optim_params = {
                "optim": "apollo_adamw",
                "optim_target_modules": [r".*\.lora_[AB]\.default$"], # we need to match the nn module here, not the parameter
                "optim_args": "rank=1,scale=128.0,scale_type=tensor", # apollo mini
            }
        else:
            optim_params = {
                "optim": "adamw_torch"
            }
    else:
        logger.info(f"Fine-tune {model_name} ({model_type}) weights")
        for name, module in model.named_modules():
            # Match exact layer name (like "transformer.h.0.attn.q_proj")
            for target_key in args.lora_target_modules.split(","):
                if name.endswith(target_key) and isinstance(module, nn.Linear):
                    logger.debug(f"training parameter {name}")
                    module.weight.requires_grad = True
                    break
        if args.apollo_adamw:
            # we use Apollo AdamW optimizer to save memory
            optim_params = {
                "optim": "apollo_adamw",
                "optim_target_modules": [r".*._proj.*"], # TODO: do we need to change this according to config.target_modules?
                "optim_args": "rank=256,scale=1.0,scale_type=channel", # apollo
            }
        else:
            optim_params = {
                "optim": "adamw_torch"
            }

    callbacks = []
    if args.uniql:
        logger.info("Using uniqlw one-pass fine-tuning")
        layer_ratio_config_dict = get_layer_ratio_config(model_name, eps=0.1)
        # Set masked_forward as the default forward function
        for name, module in model.named_modules():
            if hasattr(module, 'masked_forward'):
                module.forward = module.masked_forward
        # Add the random layer ratio callback for uniql training
        model_helper = ModelHelperRegistry[model_type](model_config=model.config)
        if isinstance(model, PeftModelForCausalLM):
            # PeftModelForCausalLM (model) -> LoraModel (base_model) -> LlamaForCausalLM (model)
            random_ratio_callback = RandomLayerRatioCallback(model.base_model.model, model_helper, layer_ratio_config_dict)
        else:
            random_ratio_callback = RandomLayerRatioCallback(model, model_helper, layer_ratio_config_dict)
        callbacks.append(random_ratio_callback)
        logger.info("Added RandomLayerRatioCallback for dynamic ratio updates during training")

    # print debugging information
    if optim_params["optim"] == "apollo_adamw":
        # we use Apollo AdamW optimizer to save memory
        logger.info("Using Apollo AdamW optimizer")
        print_matched_layers(model, optim_params["optim_target_modules"])
    print_trainable_parameters(model)

    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated before training: {max_mem:.2f} GB")
    
    # set output dir
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    if args.lora and args.uniql:
        output_dir = os.path.join(current_file_dir, "outputs", model_name, "masked-lora-rft")
    elif args.lora:
        output_dir = os.path.join(current_file_dir, "outputs", model_name, "lora-rft")
    elif args.uniql:
        output_dir = os.path.join(current_file_dir, "outputs", model_name, "masked-rft")
    else:
        output_dir = os.path.join(current_file_dir, "outputs", model_name, "rft")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Training checkpoints will be saved to {output_dir}")

    # start training
    model.train()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_loader,
        eval_dataset=valid_loader,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            # max_steps=10, # debugging only
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            save_strategy="steps",
            save_safetensors=False,
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            logging_dir=f"{output_dir}/logs",  # where tensorboard logs will be saved
            report_to="tensorboard",
            run_name=f"{model_name}-rft",
            metric_for_best_model="eval_loss",
            **optim_params,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False
    args.resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.lora:
        model = model.merge_and_unload()

    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated after training: {max_mem:.2f} GB")

    if args.quick_eval:
        from eval_utils import evaluate_ppl, eval_func
        model.eval()
        logger.info("Quickly evaluating (0-shot) the compressed model on arc_easy dataset...")
        lm_eval_results = eval_func(
            model, tokenizer, 
            model_type=model_type,
            batch_size=16,
            max_length=2048, # default
            task_list=["arc_easy"], 
            # task_list=["lambada_openai"], 
            # task_list=["arc_easy", "lambada_openai"], 
            # task_list=["hellaswag","arc_easy","arc_challenge","piqa","winogrande", "lambada_openai"], 
            fewshot=0,
            # limit=100
            limit=None
        )
    
    # store the model
    if args.pretrained_dir:
        if args.lora and args.uniql:
            model_name = get_saved_model_name(args.model_repo, postfixs=["lora-rft-uniql"])
        elif args.lora:
            model_name = get_saved_model_name(args.model_repo, postfixs=["lora-rft"])
        elif args.uniql:
            model_name = get_saved_model_name(args.model_repo, postfixs=["rft-uniql"])
        else:
            model_name = get_saved_model_name(args.model_repo, postfixs=["rft"])
        model_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        if os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} already exists, will remove it first.")
            shutil.rmtree(model_dir)  # removes the directory and its contents
        logger.info(f"Store model to {model_dir}...")
        model.save_pretrained(model_dir, max_shard_size="4GB")
        tokenizer.save_pretrained(model_dir)
    else:
        logger.warning("Not storing the fine-tuned (or lora-merged) model to --pretrained_dir, as --pretrained_dir is not provided.")
        logger.warning("Fine-tuned (or lora-merged) model not saved.")

if __name__ == "__main__":

    set_deterministic(1234)
    parser = argparse.ArgumentParser(description="Recovery fine-tuning (RFT) compressed models")
    parser = argparse_shared_options(parser)
    parser = argparse_calibration_options(parser)
    parser = argparse_compress_options(parser)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training  (default: 32)")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                        help="Micro batch size on each device for training and evaluation (default: 4)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer (default: 1e-4)")
    parser.add_argument("--apollo_adamw", action='store_true', help='use Apollo AdamW optimizer')

    # Lora Configuration
    parser.add_argument('--lora', action='store_true', help='use lora')
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    # parser.add_argument('--lora_target_modules', type=str, default=
    #                     "q_proj,k_proj,v_proj,o_proj,in_proj,gate_proj,up_proj,down_proj", help='lora target modules')
    # https://github.com/huggingface/peft/issues/2556, mamba out_proj is excluded by default. We comment out the assertion in the peft library.
    parser.add_argument('--lora_target_modules', type=str, default=
                        "q_proj,k_proj,v_proj,o_proj,in_proj,out_proj,gate_proj,up_proj,down_proj", help='lora target modules')
    parser.add_argument('--uniql', action='store_true', help='use uniql single-pass fine-tuning')

    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    # run
    main(args)