import os
import re
import copy
import json
import shutil
import logging
import argparse
from tqdm import tqdm
from functools import partial

import torch

from utils.model_utils import profile_size

from modeling.model_helper_registry import ModelHelperRegistry
from modeling.build_models import build_model_and_tokenizer

from utils.input_catcher import Catcher, CatcherExit
from utils.reproduce_utils import set_deterministic
from utils.model_utils import get_saved_model_name
from utils.model_utils import model_name_and_type
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_compress_options
from utils.args_utils import argparse_calibration_options
from utils.dataset_utils import build_dataloader
from utils.dataset_utils import get_loaders
from utils.logger_utils import set_logger

logger = logging.getLogger(os.path.basename(__file__))

@torch.no_grad()
def compress_model(model, model_type, low_rank_ratio, dataloader, nsamples=256, seqlen=1024, cache_dtype=torch.bfloat16):

    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    use_cache_orig = model.config.use_cache
    model.config.use_cache = False

    # Inference is performed in a layer-wise manner
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

    # apply hooks for layers
    act_cache = {
        "down_proj_input": [], # collect hidden_states to the down_proj
        "BC_proj_input": [], # collect hidden_states to the B_proj, C_proj
        "out_proj_input": [], # collect y_normed to the out_proj
        "q_proj_input": [], # collect hidden_states to the q_proj, and shared by k_proj and v_proj
    }

    # we can only hook the B_proj, and reuse it for C_proj
    # https://github.com/huggingface/transformers/issues/29247
    @torch.no_grad()
    def hook(module, x, y, act_cache, key, device=None):
        if type(x) is tuple:
            x = x[0] # x[0] is input hidden_states to the layer
        if device is not None:
            x = x.to(device)
        act_cache[key].append(x)

    handles = []
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        dtype = next(iter(layer.parameters())).dtype
        for name, module in layer.named_children():
            class_name = module.__class__.__name__
            if class_name in [model_helper.mlp_class_name]:
                handles.append(module.down_proj.register_forward_hook(partial(hook, act_cache=act_cache, key="down_proj_input", device="cpu")))
            elif class_name in [model_helper.attn_class_name]:
                handles.append(module.q_proj.register_forward_hook(partial(hook, act_cache=act_cache, key="q_proj_input", device="cpu")))
            elif class_name in [model_helper.mamba_simple_class_name]:
                handles.append(module.B_proj.register_forward_hook(partial(hook, act_cache=act_cache, key="BC_proj_input", device="cpu")))
                handles.append(module.out_proj.register_forward_hook(partial(hook, act_cache=act_cache, key="out_proj_input", device="cpu")))            
            elif class_name in [model_helper.mamba_class_name]:
                raise NotImplementedError("Mamba block has to be converted to the simple block for compression")
            else:
                pass
                # logger.debug(f"Not supported {class_name} compression in {model_type} layer")

    # run inferece layer by layer
    model = model.to("cpu") # move the model to the cpu to save memory
    for i in tqdm(range(len(layers))):
        assert not torch.isnan(inps).any(), f"layer {i} has nan in input"
        layer = layers[i].to(device)
        layer_ratio = low_rank_ratio[i]

        # We collect input for the next layer first, since we will make change on this layer later
        # we do it sample by sample to save memory
        dtype = next(iter(layer.parameters())).dtype
        layer = layer.to(cache_dtype) # use float32 or bfloat16 for numerical stability
        for j in range(nsamples):
            (h, ) = layer(hidden_states=inps[j].unsqueeze(0), **cached_kwargs[j])
            inps[j] = h.squeeze(0)  # store the output back to the buffer for the next layer
        layer = layer.to(dtype)
        torch.cuda.empty_cache()

        for name, module in layer.named_children():
            class_name = module.__class__.__name__

            if class_name in [model_helper.mlp_class_name]:
                down_x = torch.cat(act_cache["down_proj_input"]).to(device)
                act_cache["down_proj_input"] = []
                torch.cuda.empty_cache()
                logger.debug(f"Layer {i} mlp ratio: {layer_ratio['mlp_ratio']}")
                model_helper.compress_mlp(layer, down_x, layer_ratio['mlp_ratio'])
                del down_x
                torch.cuda.empty_cache()
            elif class_name in [model_helper.attn_class_name]:
                qkv_x = torch.cat(act_cache["q_proj_input"]).to(device)
                act_cache["q_proj_input"] = []
                torch.cuda.empty_cache()
                logger.debug(f"Layer {i} attn ratio: {layer_ratio['attn_ratio']}")
                model_helper.compress_attn(layer, qkv_x, layer_ratio['attn_ratio'], cached_kwargs[0].get("position_embeddings", None))
                del qkv_x
                torch.cuda.empty_cache()
            elif class_name in [model_helper.mamba_simple_class_name]:
                hidden_states = torch.cat(act_cache["BC_proj_input"]).to(device)
                y_normed = torch.cat(act_cache["out_proj_input"]).to(device)
                act_cache["BC_proj_input"] = []
                act_cache["out_proj_input"]  = []
                torch.cuda.empty_cache()
                logger.debug(f"Layer {i} mamba ratio: {layer_ratio['mamba_ratio']}")
                model_helper.compress_mamba(layer, hidden_states, y_normed, layer_ratio['mamba_ratio'])
                del hidden_states
                del y_normed
                torch.cuda.empty_cache()
            elif class_name in [model_helper.mamba_class_name]:
                raise NotImplementedError("Mamba block has to be converted to the simple block for compression")
            else:
                pass
                # logger.debug(f"Not supported {class_name} compression")
        layer = layer.cpu() # temporary move the compressed layer to the cpu to save memory  
        torch.cuda.empty_cache()
    
    del act_cache
    torch.cuda.empty_cache()
    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated: {max_mem:.2f} GB")
    # copy the compress config to the model config
    model.config.compress_config = copy.deepcopy(model_helper.config.compress_config)
    # Remove hooks
    for h in handles:
        h.remove()
    
    # move model back to the device at the end
    model = model.to(device)
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


def configure_model(model, model_type):
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    # replace the layer to simple versions for collecting the scaling factors
    layers = model_helper.get_layers(model)
    for layer_idx, layer in enumerate(tqdm(layers, desc=f"Replacing {model_type} layers for compression")):
        for name, module in layer.named_children():
            class_name = module.__class__.__name__
            if class_name in [model_helper.mlp_class_name]:
                pass
            elif class_name in [model_helper.attn_class_name]:
                pass
            elif class_name in [model_helper.mamba_class_name]:
                # we use a simple mamba block to make things easier
                logger.debug(f"Replace {class_name} with simple mamba block")
                # mamba = model_helper.get_mamba(layer)
                # fuse_ln_linear(mamba.norm, mamba.out_proj) # must fuse layernorm to the out_proj for Nemotron-H
                model_helper.replace_simple_mamba(layer, use_had_transform=False)
            else:
                pass
    model.eval()
    return model


def main(args):

    if not args.pretrained_dir:
        logger.warning("--pretrained_dir is not provided.")
        
    logger.info(f"Compress {args.model_repo}")
    logger.info(f"Use dataset {args.calib_data_repo}")
    logger.info(f"Number of samples: {args.calib_data_num}")
    logger.info(f"Sequence length: {args.calib_seqlen}")

    model, tokenizer = build_model_and_tokenizer(args.model_repo, pretrained_dir=args.pretrained_dir, dtype=torch.bfloat16)
    model_name, model_type = model_name_and_type(args.model_repo)
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
    
    # apply layer-wise compression config
    if args.layer_ratio_config is not None:
        logger.info(f"Loading layer-wise compression config {args.layer_ratio_config}...")
        with open(args.layer_ratio_config, "r") as f:
            layer_ratio = json.load(f)
        for i, d in enumerate(layer_ratio):
            logger.debug(f"Layer {i}: {d}")
        match = re.search(r'ratio-([0-9]+(?:\.[0-9]+)?)', args.layer_ratio_config)
        ratio = float(match.group(1))
        logger.info(f"Get average ratio: {ratio}")
    else:
        if args.uniform_ratio == 1.0:
            logger.warning("Uniform ratio is 1.0. Applying weight sorting instead of compression")
        else:
            logger.info(f"Use uniform ratio {args.uniform_ratio} for the compressing the model")
        ratio = float(args.uniform_ratio)
        model_helper = ModelHelperRegistry[model_type](model_config=model.config)
        layer_ratio = []
        layers = model_helper.get_layers(model)
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            layer_ratio_dict = {}
            for name, module in layer.named_children():
                class_name = module.__class__.__name__
                if class_name in [model_helper.mlp_class_name]:
                    layer_ratio_dict["mlp_ratio"] = args.uniform_ratio
                elif class_name in [model_helper.attn_class_name]:
                    layer_ratio_dict["attn_ratio"] = args.uniform_ratio
                elif class_name in [model_helper.mamba_class_name]:
                    layer_ratio_dict["mamba_ratio"] = args.uniform_ratio
                else:
                    pass
            layer_ratio.append(layer_ratio_dict)
    # configure the model
    model = configure_model(model, model_type)

    # compress the model
    model = compress_model(model, model_type, low_rank_ratio=layer_ratio, dataloader=dataloader,
                           nsamples=args.calib_data_num, seqlen=args.calib_seqlen, cache_dtype=torch.float32)

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
            fewshot=0,
            limit=None
        )
    
    # store the model
    if args.pretrained_dir:
        model_name = get_saved_model_name(args.model_repo, postfixs=[f"uniql-{ratio}"])
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
    parser = argparse.ArgumentParser(description="Compressing models")
    parser = argparse_shared_options(parser)
    parser = argparse_calibration_options(parser)
    parser = argparse_compress_options(parser)
    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    # run
    main(args)
