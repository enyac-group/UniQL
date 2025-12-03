import gc
import os
import re
import json
import copy
import shutil
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.model_utils import profile_size

from modeling.model_helper_registry import ModelHelperRegistry
from modeling.build_models import build_model_and_tokenizer


from quantize.gptq import GPTQ
from quantize.quant_embedding_layers import W4O16Embedding
from quantize.quant_linear_layers import W4A16B16O16Linear
from quantize.quant_linear_layers import HadLinear

from quantize.hadamard_helpers import had_transform

from utils.input_catcher import Catcher, CatcherExit
from utils.logger_utils import set_logger
from utils.dataset_utils import get_loaders
from utils.dataset_utils import build_dataloader
from utils.model_utils import model_name_and_type
from utils.model_utils import get_saved_model_name
from utils.reproduce_utils import set_deterministic
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_compress_options
from utils.args_utils import argparse_quantize_options
from utils.args_utils import argparse_calibration_options


logger = logging.getLogger(os.path.basename(__file__))


@torch.no_grad()
def apply_gptq(model, model_type, w_bits, dataloader, nsamples=256, seqlen=1024, cache_dtype=torch.float32):
    # 2-bit and 3-bit is only for simulation, we will run them with w4a16 kernels
    assert w_bits in [2, 3, 4], "Only support 2/3/4-bit weights for now"
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    use_cache_orig = model.config.use_cache
    model.config.use_cache = False

    # Quantize embedding first, where we always apply 4-bit quantization
    embeddings = model_helper.get_embeddings(model)
    embed_dtype = embeddings.weight.dtype
    embed_weight = embeddings.weight.data
    max_val = 16 # 4-bit max_val = 16
    zero_point = 8 # 4-bit zero_point = 8
    token_scale = embed_weight.abs().amax(dim=-1, keepdim=True) / (max_val / 2)
    embed_weight_4bit = torch.round((embed_weight / token_scale) + zero_point).clamp(0, max_val-1)
    embeddings.weight.data = ((embed_weight_4bit - zero_point) * token_scale).to(embed_dtype)
    
    # Here, inference are performed in a layer-wise manner
    layers = model_helper.get_layers(model)
    dtype = next(iter(model.parameters())).dtype
    device = next(iter(model.parameters())).device
    # cache the input of the first layer
    layers[0] = Catcher(layers[0].to(cache_dtype)) # use float32 or bfloat16 for numerical stability
    layers[0].init_input_buffer(nsamples, seqlen, model.config.hidden_size, dtype=cache_dtype, device=device)
    for batch in dataloader:
        if type(batch) is tuple: # FIXME: in order to use self-implemented wikitext2
            input_ids = batch[0].to(device)
        else:
            input_ids = batch["input_ids"].to(device)
        try:
            model(input_ids=input_ids)
        except CatcherExit:
            pass
    # get inputs, restore layer, and free memory 
    inps = layers[0].input_buffer
    cached_kwargs = layers[0].cached_kwargs
    layers[0] = layers[0].module.to(dtype)
    torch.cuda.empty_cache()

    # run GPTQ in a layer-wise manner to quantize the layers to w_bits
    for i in tqdm(range(len(layers))):
        assert not torch.isnan(inps).any(), f"layer {i} has nan in input"
        layer = layers[i]
        # create GPTQ objects and register hooks for GPTQ add_batch
        gptq = {}
        handles = []
        for name, module in layer.named_modules():
            if isinstance(module, (nn.Linear, HadLinear)):
                logger.debug(f"Registering GPTQ for layer.{i}.{name}: {module}")
                gptq[name] = GPTQ(module, dtype=torch.float32)
                handle = module.register_forward_hook(gptq[name].add_batch)
                handles.append(handle)
            else:
                pass
        # collect inputs for GPTQ operators, we run it sample by sample to save memory for gptq.add_batch
        dtype = next(iter(layer.parameters())).dtype
        layer = layer.to(cache_dtype) # use float32 or bfloat16 for numerical stability
        for j in range(nsamples):
            (h, ) = layer(hidden_states=inps[j].unsqueeze(0), **cached_kwargs[j])
            # inps[j] = h.squeeze(0)  # store the output back to the buffer for the next layer
        layer = layer.to(dtype)
        # remove the gptq.add_batch hooks
        for h in handles:
            h.remove()
        # start running GPTQ for operators, and free the memory at the end
        for name in gptq.keys():
            logger.debug(f"Performing GPTQ on layer.{i}.{name} with {w_bits} bits")
            # hardcode group size to 128
            gptq[name].fasterquant(percdamp=0.01, group_size=128, w_bits=w_bits)
            gptq[name].free()
        del gptq
        # Use the GPTQ corrected layer to run again, and get the input for the next layer
        dtype = next(iter(layer.parameters())).dtype
        layer = layer.to(cache_dtype) # use float32 or bfloat16 for numerical stability
        for j in range(nsamples):
            (h, ) = layer(hidden_states=inps[j].unsqueeze(0), **cached_kwargs[j])
            inps[j] = h.squeeze(0)  # store the output back to the buffer for the next layer
        layer = layer.to(dtype)
        # clean up
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Quantizing lm_head with GPTQ")
    # final norm
    final_layernorm = model_helper.get_final_layernorm(model)
    final_layernorm = final_layernorm.to(cache_dtype)  # use float32 or bfloat16 for numerical stability
    # we run it sample by sample to save memory
    for j in range(nsamples):
        inps[j] = final_layernorm(inps[j].unsqueeze(0))
    final_layernorm = final_layernorm.to(dtype)
    # lm_head
    lm_head = model_helper.get_lm_head(model)
    logger.debug(f"Registering GPTQ for lm_head: {lm_head}")
    gptq_lm_head = GPTQ(lm_head, dtype=torch.float32)
    handle = lm_head.register_forward_hook(gptq_lm_head.add_batch)
    # we run it sample by sample to save memory for gptq.add_batch
    dtype = next(iter(lm_head.parameters())).dtype
    lm_head = lm_head.to(cache_dtype) # use float32 or bfloat16 for numerical stability
    for j in range(nsamples):
        lm_head(inps[j].unsqueeze(0))
    lm_head = lm_head.to(dtype)
    handle.remove()
    # run GPTQ to quantize lm_head to w_bits
    logger.debug(f"Performing GPTQ on lm_head with {w_bits} bits")
    gptq_lm_head.fasterquant(percdamp=0.01, group_size=128, w_bits=w_bits)
    gptq_lm_head.free()
    del gptq_lm_head
    torch.cuda.empty_cache()
    gc.collect()

    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated: {max_mem:.2f} GB")
    return model


@torch.no_grad()
def quantize_fp16_model(model, model_type, w_bits=4):

    assert w_bits in [4, ], "Only support 4 bits weights for now"
    model.config.use_cache = False
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)

    # replace embedding layer, only support w4a16 quantization for now
    logger.info(f'Applying quantized embedding')
    embed = model_helper.get_embeddings(model)
    qembed = W4O16Embedding.from_fp16(embed)
    model_helper.set_embeddings(model, qembed)
    gc.collect()
    torch.cuda.empty_cache()

    # replace layers, only support w4a16 quantization for now
    layers = model_helper.get_layers(model)
    for layer_idx, layer in enumerate(tqdm(layers, desc=f"Applying quantized layers, model type: {model_type}")):
        for name, module in layer.named_children():
            class_name = module.__class__.__name__
            if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name, model_helper.mlp_simple_class_name]:
                model_helper.replace_w4a16_mlp(layer)
            elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name, model_helper.attn_simple_class_name]:
                model_helper.replace_w4a16_attn(layer)
            elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name, model_helper.mamba_simple_class_name]:
                model_helper.replace_w4a16_mamba(layer)
            else:
                logger.debug(f"{class_name} in {model_type} layer is not supported for w4a16 quantization")

    # replace lm_head, only support w4a16 quantization for now
    logger.info(f'Applying quantized lm_head')
    head = model_helper.get_lm_head(model)
    qhead = W4A16B16O16Linear.from_fp16(head)
    model_helper.set_lm_head(model, qhead)
    gc.collect()
    torch.cuda.empty_cache()
    
    model.eval()
    return model


@torch.no_grad()
def fuse_ln_linear(norm, linear) -> None:
    """
    fuse the layernorm weight to the adjacent linear layer.
    """

    # Calculating new weight and bias
    if hasattr(linear, 'weight'):
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)  
        if hasattr(norm, 'bias') and norm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
            linear.bias.data = linear.bias.data.to(linear_dtype)
    else:
        raise ValueError(f"Unknown linear type {linear.__class__.__name__}")
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1

@torch.no_grad()
def fuse_ln_mlp_layer(norm, mlp):
    fuse_ln_linear(copy.deepcopy(norm), mlp.up_proj)
    if hasattr(mlp, 'gate_proj'):
        fuse_ln_linear(copy.deepcopy(norm), mlp.gate_proj)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1

@torch.no_grad()
def fuse_ln_moe_layer(norm, moe):
    fuse_ln_linear(copy.deepcopy(norm), moe.router)
    for i in range(moe.num_experts):
        fuse_ln_mlp_layer(copy.deepcopy(norm), moe.experts[i])
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1

@torch.no_grad()
def fuse_ln_attn_layer(norm, attn):
    fuse_ln_linear(copy.deepcopy(norm), attn.q_proj)
    fuse_ln_linear(copy.deepcopy(norm), attn.k_proj)
    fuse_ln_linear(copy.deepcopy(norm), attn.v_proj)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1


@torch.no_grad()
def fuse_had_matrices(model, model_type):
    # get model helper
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    layers = model_helper.get_layers(model)

    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    for i in range(len(layers)):
        for name, module in layers[i].named_modules():
            if isinstance(module, (HadLinear)):
                logger.debug(f"Fusing Hadamard matrices to weights for layer.{i}.{name}")
                module.fuse_hadamard()
    return model


def configure_model(model, model_type, use_had_transform=False):

    logger.info(f"Configuring {model_type} model, use_had_transform: {use_had_transform}")

    # get model helper
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)

    # apply hadamard transform to the embeddings and lm_head
    if use_had_transform:
        embeddings = model_helper.get_embeddings(model)
        final_layernorm = model_helper.get_final_layernorm(model)
        lm_head = model_helper.get_lm_head(model)
        if embeddings is lm_head: # if lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_weight = embeddings.weight.data.clone()
        else:
            lm_head_weight = lm_head.weight.data
        embeddings.weight.data = had_transform(embeddings.weight.data)  # transform embedding first
        lm_head.weight = torch.nn.Parameter(lm_head_weight * final_layernorm.weight.view(1, -1)) # fuse layernorm first, must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
        final_layernorm.weight.data = torch.ones_like(final_layernorm.weight) # fuse layernorm first
        lm_head.weight.data = had_transform(lm_head.weight.data) # and then transform the lm_head
        torch.cuda.empty_cache()
        gc.collect()

    # replace the layer to simple versions for collecting the scaling factors
    layers = model_helper.get_layers(model)
    for layer_idx, layer in enumerate(tqdm(layers, desc=f"Replacing {model_type} modules for quantization")):
        for name, module in layer.named_children():
            class_name = module.__class__.__name__
            if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                if use_had_transform:
                    norm = model_helper.get_mlp_norm(layer)
                    mlp = model_helper.get_mlp(layer)
                    fuse_ln_mlp_layer(norm, mlp)
                model_helper.replace_simple_mlp(layer, use_had_transform=use_had_transform)
            elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                if use_had_transform:
                    norm = model_helper.get_attn_norm(layer)
                    attn = model_helper.get_attn(layer)
                    fuse_ln_attn_layer(norm, attn)
                model_helper.replace_simple_attn(layer, use_had_transform=use_had_transform)
            elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                if use_had_transform:
                    norm = model_helper.get_mamba_norm(layer)
                    mamba = model_helper.get_mamba(layer)
                    fuse_ln_linear(norm, mamba.in_proj)
                model_helper.replace_simple_mamba(layer, use_had_transform=use_had_transform)
            else:
                logger.debug(f"layer type {class_name} is not configured for quantization")
    model = fuse_had_matrices(model, model_type)
    model.eval()
    # logger.debug(model)
    return model


def main(args):

    if not args.pretrained_dir:
        logger.warning("--pretrained_dir is not provided.")

    logger.info(f"Apply Quantization to {args.model_repo}")

    # apply layer-wise compression config
    layer_ratio = None
    ratio = None

    if args.layer_ratio_config is not None:
        logger.info(f"Loading layer-wise compression config {args.layer_ratio_config}...")
        with open(args.layer_ratio_config, "r") as f:
            layer_ratio = json.load(f)
        for i, r in enumerate(layer_ratio):
            logger.debug(f"Layer {i}: {r:.2f}")
        match = re.search(r'ratio-([0-9]+(?:\.[0-9]+)?)', args.layer_ratio_config)
        ratio = float(match.group(1))
        logger.info(f"Get average ratio: {ratio}")

    model_name, model_type = model_name_and_type(args.model_repo)
    # torch.bfloat16 model will not work for quantization, but we need this to make Nemotron-H-8B-Base-8K work for gtpq
    model, tokenizer = build_model_and_tokenizer(model_repo=args.model_repo, pretrained_dir=args.pretrained_dir,
                                                 layer_ratio_config=layer_ratio, dtype=torch.bfloat16)
    if model_type == "qwen2.5" and args.hadamard:
        logger.warning(f"Hadamard transform is not supported for {model_type}, will set hadamard to False.")
        args.hadamard = False
    model = configure_model(model, model_type, use_had_transform=args.hadamard)

    # update config
    if args.layer_ratio_config is not None:
        reduced_head_dim_config = []
        reduced_intermediate_size_config = []
        model_helper = ModelHelperRegistry[model_type](model_config=model.config)
        layers = model_helper.get_layers(model)
        for layer_idx, layer in enumerate(layers):
            # get reduced_head_dim and reduced_intermediate_size for this layer
            reduced_head_dim_config.append(layer.self_attn.head_dim)
            reduced_intermediate_size_config.append(layer.mlp.intermediate_size)
        # FIXME: this does not work for Bamba and Nemotron-H
        model.config.compress_config = {
            "reduced_head_dim": reduced_head_dim_config,
            "reduced_intermediate_size": reduced_intermediate_size_config,
        }
    
    # FIXME: gptq does not work for nvidia/Nemotron-H-8B-Base-8K
    if args.gptq:
        logger.info(f"Applying {args.w_bits}-bit weights GPTQ on {model_name} ({model_type})")
        logger.info(f"Use dataset {args.calib_data_repo}")
        logger.info(f"Number of samples: {args.calib_data_num}")
        logger.info(f"Sequence length: {args.calib_seqlen}")
        if args.calib_data_repo == "wikitext2":
            # we choose to use most common calibration set from wiki2 in the most papers
            logger.debug("Use self-implemented wikitext2 as the calibration set")
            dataloader, _ = get_loaders("wikitext2", tokenizer, args.calib_data_num, 1234, args.calib_seqlen, model)
        else:
            dataloader, _ = build_dataloader(
                args.calib_data_repo, tokenizer, batch_size=1,
                num_sample=args.calib_data_num, max_length=args.calib_seqlen,
                enable_instruct_prompting=False, instruct_output_length=args.calib_seqlen,
                columns=["input_ids"])
        # we need to use float32 to make Nemotron-H-8B-Base-8K work for gtpq
        model = apply_gptq(model, model_type, w_bits=args.w_bits, dataloader=dataloader,
                        nsamples=args.calib_data_num, seqlen=args.calib_seqlen, cache_dtype=torch.float32)

        

    if args.w_bits == 4:
        logger.info(f"Replacing fp16 blocks with 4-bit blocks")
        dtype = next(iter(model.parameters())).dtype
        if dtype is not torch.float16:
            logger.warning(f"Converting model to float16 before applying 4-bit blocks")
            model = model.to(torch.float16)
        model = quantize_fp16_model(model, model_type, w_bits=4)
    else:
        logger.error(f"Not supported {args.w_bits}-bit real quantization for now.")
        exit(1)

    # get model size
    profile_size(model, model_type, batch_size=1, prompt_len=1024)
    torch.cuda.empty_cache()

    if args.quick_eval:
        from eval_utils import evaluate_ppl, eval_func
        logger.info("Quickly evaluating (0-shot) the compressed model on arc_easy dataset...")
        lm_eval_results = eval_func(
            model, tokenizer, 
            model_type=model_type,
            batch_size=16,
            max_length=2048, # default
            task_list=["arc_easy"], 
            # task_list=["hellaswag","arc_easy","arc_challenge","piqa","winogrande"], 
            fewshot=0,
            # limit=100
            limit=None
        )
    
    # store the model
    if args.pretrained_dir:
        postfixs = f"w{args.w_bits}a16-{ratio:.2f}" if ratio else f"w{args.w_bits}a16"
        model_name = get_saved_model_name(args.model_repo, postfixs=[postfixs])
        model_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        if os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} already exists, will remove it first.")
            shutil.rmtree(model_dir)  # removes the directory and its contents
        logger.info(f"Store model to {model_dir}...")
        model.save_pretrained(model_dir, max_shard_size="4GB")
        tokenizer.save_pretrained(model_dir)
    else:
        logger.warning("Not storing the model, as --pretrained_dir is not provided.")
        logger.warning("Model not saved.")

if __name__ == "__main__":

    set_deterministic(1234)
    parser = argparse.ArgumentParser(description="Quantize model with GPTQ")
    parser = argparse_shared_options(parser)
    parser = argparse_calibration_options(parser)
    parser = argparse_compress_options(parser)
    parser = argparse_quantize_options(parser)
    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    # run
    main(args)