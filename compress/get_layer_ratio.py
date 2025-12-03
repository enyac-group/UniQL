import os
import json
import logging
import argparse
from tqdm import tqdm

import torch

from modeling.model_helper_registry import ModelHelperRegistry
from modeling.build_models import build_model_and_tokenizer

from utils.input_catcher import Catcher, CatcherExit
from utils.logger_utils import set_logger
from utils.dataset_utils import get_loaders
from utils.dataset_utils import build_dataloader
from utils.model_utils import model_name_and_type
from utils.reproduce_utils import set_deterministic
from utils.args_utils import argparse_shared_options
from utils.args_utils import argparse_calibration_options
from utils.args_utils import argparse_compress_options


logger = logging.getLogger(os.path.basename(__file__))


def allocate_layerwise_sparsity(scores, phi_avg: float, epsilon: float = 1, weights: list = None):
    """
    Allocate sparsity per layer using entropic regularization.

    Args:
        scores (Tensor or list): Importance scores, shape [L]
        phi_avg (float): Target global sparsity (e.g., 0.3)
        epsilon (float): Entropic regularization strength
        weights (Tensor or list): Weights, shape [L]

    Returns:
        phi (Tensor): Per-layer sparsity allocations, shape [L]
    """
    saved_type = type(scores)
    if saved_type is list:
        scores = torch.tensor(scores)
        if weights is None:
            weights = torch.ones_like(scores)
        else:
            weights = torch.tensor(weights)

    L = scores.numel()
    scaled_scores = -scores / epsilon
    phi = (L * phi_avg) * (torch.softmax(weights * scaled_scores, dim=0))
    # w = torch.unique(weights).sum()
    # phi = phi * w
    if phi.min() < 0:
        logger.warning(f"phi has negative values: {phi.min()} for compression rate {phi_avg}. The smallest value will be set to 0")
    if phi.max() > 1:
        logger.warning(f"phi has values greater than 1: {phi.max()} for compression rate {phi_avg}. The largest value will be set to 0.85")
    if phi.max() > 0.85:
        logger.warning(f"phi has values greater than 0.85: {phi.max()} for compression rate {phi_avg}. The largest value will be set to 0.85")

    phi = phi.clamp(0, 0.85) # This means the we keep at most 100% at least 15% dimension of the layer

    if saved_type is list:
        phi = phi.tolist()
    return phi


def compute_layer_importance(x: torch.Tensor, y: torch.Tensor, final_token_only=True) -> float:
    """
    Compute the importance score s = 1 - cos(x, y), averaged over all tokens.

    Args:
        x (Tensor): [B, N, D]
        y (Tensor): [B, N, D]

    Returns:
        importance (float): scalar importance score in [0, 1]
    """

    if final_token_only:
        # use only last token for angular distance as described in section 3.2
        # https://arxiv.org/pdf/2403.17887.pdf
        # "due to the causal attention mask, its embedding is the only one that depends on the entire sequence"
        x = x[:, -1:, :]
        y = y[:, -1:, :]

    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])

    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
    
    sim = (x * y).sum(dim=-1)  # cosine similarity [-1, 1]
    importance = 1 - sim.mean().item()  # cosine distance [0, 2] == importance
    return importance
    # return importance / 2. # Normalize to [0, 1] range


@torch.no_grad()
def get_layerwise_importance(model, model_type, dataloader, nsamples=256, seqlen=1024, fine_grained=False):

    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    use_cache_orig = model.config.use_cache
    model.config.use_cache = False

    # Here, inference are performed in a layer-wise manner
    layers = model_helper.get_layers(model)
    dtype = next(iter(model.parameters())).dtype
    device = next(iter(model.parameters())).device
    # cache the input of the first layer
    layers[0] = Catcher(layers[0].to(torch.float32))
    layers[0].init_input_buffer(nsamples, seqlen, model.config.hidden_size, dtype=torch.float32, device=device)
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

    # get layer importance in a layer-wise manner
    layer_importance = []
    layer_weights = []
    saved_blocks = []
    for i in tqdm(range(len(layers))):
        layer = layers[i]

        @torch.no_grad()
        def input_hook(module, x, y):
            if type(x) is tuple:
                x = x[0]
            if not hasattr(module, "inputs"):
                module.inputs = []
            module.inputs.append(x.detach().cpu())

        @torch.no_grad()
        def output_hook(module, x, y):
            if type(y) is tuple:
                y = y[0] # attention returns a tuple
            if not hasattr(module, "outputs"):
                module.outputs = []
            module.outputs.append(y.detach().cpu())

        if fine_grained:
            # https://github.com/huggingface/transformers/issues/29247
            handles = []
            for name, module in layer.named_children():
                class_name = module.__class__.__name__
                if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                    handles.append(model_helper.get_mlp_norm(layer).register_forward_hook(input_hook))
                    handles.append(model_helper.get_mlp(layer).register_forward_hook(output_hook))
                elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                    handles.append(model_helper.get_attn_norm(layer).register_forward_hook(input_hook))
                    handles.append(model_helper.get_attn(layer).register_forward_hook(output_hook))
                elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                    handles.append(model_helper.get_mamba_norm(layer).register_forward_hook(input_hook))
                    handles.append(model_helper.get_mamba(layer).register_forward_hook(output_hook))
                else:
                    # logger.debug(f"No register forward hook for {class_name} in {model_type} layer")
                    continue
        else:
            #FIXME: This work as fine-grained for Nemontron-H
            handles = [
                model_helper.get_input_layernorm(layer).register_forward_hook(input_hook),
                # layer.input_layernorm.register_forward_hook(input_hook),
                layer.register_forward_hook(output_hook),
            ]
        # We collect input for the next layer first, since we will make change on this layer later
        # we do it sample by sample to save memory
        dtype = next(iter(layer.parameters())).dtype
        layer = layer.to(torch.float32) # use float32 for numerical stability
        for j in range(nsamples):
            (h, ) = layer(hidden_states=inps[j].unsqueeze(0), **cached_kwargs[j])
            inps[j] = h.squeeze(0)  # store the output back to the buffer for the next layer
        layer = layer.to(dtype)

        # Remove hooks
        for h in handles:
            h.remove()

        if fine_grained:
            for name, module in layer.named_children():
                class_name = module.__class__.__name__
                if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                    x = torch.cat(model_helper.get_mlp_norm(layer).inputs).to(device)
                    y = torch.cat(model_helper.get_mlp(layer).outputs).to(device)
                    del model_helper.get_mlp_norm(layer).inputs
                    del model_helper.get_mlp(layer).outputs
                    layer_importance.append(compute_layer_importance(x, y))
                    layer_weights.append(2.0) # the approximation of parameter size ratio in a block
                    logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                    x = torch.cat(model_helper.get_attn_norm(layer).inputs).to(device)
                    y = torch.cat(model_helper.get_attn(layer).outputs).to(device)
                    del model_helper.get_attn_norm(layer).inputs
                    del model_helper.get_attn(layer).outputs
                    layer_importance.append(compute_layer_importance(x, y))
                    layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                    logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                    x = torch.cat(model_helper.get_mamba_norm(layer).inputs).to(device)
                    y = torch.cat(model_helper.get_mamba(layer).outputs).to(device)
                    del model_helper.get_mamba_norm(layer).inputs
                    del model_helper.get_mamba(layer).outputs
                    layer_importance.append(compute_layer_importance(x, y))
                    layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                    logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                else:
                    # logger.debug(f"No compute layer importance for {class_name} in {model_type} layer")
                    continue
                torch.cuda.empty_cache()
        else:

            if model_helper.layer_step == 1:
                #FIXME: This work as fine-grained for Nemontron-H
                x = torch.cat(model_helper.get_input_layernorm(layer).inputs).to(device)
                y = torch.cat(layer.outputs).to(device)
                del model_helper.get_input_layernorm(layer).inputs
                del layer.outputs
                torch.cuda.empty_cache()

                # get mlp layer importance here, or Block influence (BI) scores
                # https://arxiv.org/pdf/2403.03853
                # layer_importance.append(compute_layer_importance(x, y))
                s = compute_layer_importance(x, y)
                for name, module in layer.named_children():
                    class_name = module.__class__.__name__
                    if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                        layer_importance.append(s)
                        layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                        logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                    elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                        layer_importance.append(s)
                        layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                        logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                    elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                        layer_importance.append(s)
                        layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                        logger.debug(f"Layer {i} {class_name} importance: {layer_importance[-1]:.4f}")
                    else:
                        # logger.debug(f"No compute layer importance for {class_name} in {model_type} layer")
                        continue
                    torch.cuda.empty_cache()

            if model_helper.layer_step == 2:
                # [7, 18, 29, 40]
                # if i % 2 == 0: # this means mixer block in Nemotron-H
                if layer.mixer.__class__.__name__ in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                    print(i, layer.mixer.__class__.__name__)
                    x = torch.cat(model_helper.get_input_layernorm(layer).inputs).to(device)
                    del model_helper.get_input_layernorm(layer).inputs
                    del layer.outputs
                    saved_blocks.append((layer, x))
                if layer.mixer.__class__.__name__ in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                    print(i, layer.mixer.__class__.__name__)
                    del model_helper.get_input_layernorm(layer).inputs
                    del layer.outputs
                    saved_blocks.append((layer, None))
                # if i % 2 == 1: # this means mlp block in Nemotron-H
                if layer.mixer.__class__.__name__ in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                    print(i, layer.mixer.__class__.__name__)
                    y = torch.cat(layer.outputs).to(device)
                    del model_helper.get_input_layernorm(layer).inputs
                    del layer.outputs
                    saved_blocks.append((layer, y))

                    # get mlp layer importance here, or Block influence (BI) scores
                    # https://arxiv.org/pdf/2403.03853
                    # layer_importance.append(compute_layer_importance(x, y))
                    if len(saved_blocks) == 2:
                        mixer_block, x = saved_blocks[0]
                        mlp_block, y = saved_blocks[1]
                        iterator = zip([mixer_block, mlp_block], [-1, 0])
                    elif len(saved_blocks) == 3:
                        mixer_block, x = saved_blocks[0]
                        attn_block, _ = saved_blocks[1]
                        mlp_block, y = saved_blocks[2]
                        iterator = zip([mixer_block, attn_block, mlp_block], [-2, -1, 0])
                    else:
                        raise ValueError(f"Saved blocks length is not 2 or 3: {len(saved_blocks)}")
                    s = compute_layer_importance(x, y)
                    for block, offset in iterator:
                        for name, module in block.named_children():
                            class_name = module.__class__.__name__
                            if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                                layer_importance.append(s)
                                layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                                logger.debug(f"Layer {i + offset} {class_name} importance: {layer_importance[-1]:.4f}")
                            elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                                layer_importance.append(s)
                                layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                                logger.debug(f"Layer {i + offset} {class_name} importance: {layer_importance[-1]:.4f}")
                            elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                                layer_importance.append(s)
                                layer_weights.append(1.0) # the approximation of parameter size ratio in a block
                                logger.debug(f"Layer {i + offset} {class_name} importance: {layer_importance[-1]:.4f}")
                            else:
                                # logger.debug(f"No compute layer importance for {class_name} in {model_type} layer")
                                continue
                    del x, y
                    saved_blocks = []
                    torch.cuda.empty_cache()
    # Get max memory usage in GB
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    logger.info(f"Max memory allocated: {max_mem:.2f} GB")

    return layer_importance, layer_weights



def main(args):

    model_name, model_type = model_name_and_type(args.model_repo)
    model, tokenizer = build_model_and_tokenizer(args.model_repo, pretrained_dir=args.pretrained_dir, dtype=torch.float16)

    if not args.fine_grained and model_type == "nemotron_h":
        logger.warning("Nemotron-H only support fine-grained mode for now")
        args.fine_grained = True

    if not args.uniform:
        logger.info(f"Compute layer-wise importance for {model_name} ({model_type})")
        logger.info(f"Use dataset {args.calib_data_repo}")
        logger.info(f"Number of samples: {args.calib_data_num}")
        logger.info(f"Sequence length: {args.calib_seqlen}")
        logger.info(f"Fine-grained mode: {args.fine_grained}")

        if args.cached_layer_importance is not None:
            with open(args.cached_layer_importance, "r") as f:
                data = json.load(f)
            layer_importance = data["layer_importance"]
            layer_weights = data["layer_weights"]
        else:
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

            layer_importance, layer_weights = get_layerwise_importance(model, model_type, dataloader=dataloader,
                                                        nsamples=args.calib_data_num, seqlen=args.calib_seqlen,
                                                        fine_grained=args.fine_grained)

            # Store the layer-wise importance
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_file_dir, "outputs", model_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dataset_name = args.calib_data_repo.split("/")[-1]
            filename = f"layer_importance_{dataset_name}"
            if args.fine_grained:
                filename = filename + "_fine_grained"
            filename = filename + ".json"
            logger.info(f"Save the layer-wise importance to {output_dir}/{filename}")
            with open(f"{output_dir}/{filename}", "w") as f:
                json.dump({"layer_importance": layer_importance, "layer_weights": layer_weights}, f)

    
    if type(args.ratio_list) is float:
        ratio_list = [args.ratio_list]
    elif type(args.ratio_list) is list:
        ratio_list = [float(r) for r in args.ratio_list]
    else:
        raise ValueError(f"Unsupported ratio_list type: {type(args.ratio_list)}")
    
    # We generate all configurations at once.
    model_helper = ModelHelperRegistry[model_type](model_config=model.config)
    layers = model_helper.get_layers(model)
    for ratio in ratio_list:
        # allocate layer-wise low-rank ratio
        if args.uniform:
            logger.info(f"Apply uniform low-rank ratio {ratio} to {model_name} ({model_type})")
            output_config = [{} for _ in range(len(layers))]
            for i in tqdm(range(len(layers))):
                layer = layers[i]
                for name, module in layer.named_children():
                    class_name = module.__class__.__name__
                    if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                        output_config[i]["mlp_ratio"] = ratio
                        logger.debug(f"Layer {i} {class_name} ratio: {ratio:.2f}")
                    elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                        output_config[i]["attn_ratio"] = ratio
                        logger.debug(f"Layer {i} {class_name} ratio: {ratio:.2f}")
                    elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                        output_config[i]["mamba_ratio"] = ratio
                        logger.debug(f"Layer {i} {class_name} ratio: {ratio:.2f}")
                    else:
                        pass
        else:
            logger.info(f"Allocate layer-wise low-rank ratio with average ratio {ratio} and epsilon {args.epsilon} for {model_name} ({model_type})")
            # 1-ratio to sparsity ratio
            layer_sparity = allocate_layerwise_sparsity(layer_importance, 1-ratio, epsilon=args.epsilon, weights=layer_weights)
            # convert to layer ratio so it is easier to create torch.nn.Linear
            # the ratio is the preserved dimension of the layer
            layer_ratio = [1 - r for r in layer_sparity]

            j = 0
            output_config = [{} for _ in range(len(layers))]
            for i in tqdm(range(len(layers))):
                layer = layers[i]
                for name, module in layer.named_children():
                    class_name = module.__class__.__name__
                    if class_name in [model_helper.mlp_class_name, model_helper.mlp_uniql_class_name]:
                        output_config[i]["mlp_ratio"] = layer_ratio[j]
                        logger.debug(f"Layer {i} {class_name} ratio: {layer_ratio[j]:.2f}")
                        j += 1
                    elif class_name in [model_helper.attn_class_name, model_helper.attn_uniql_class_name]:
                        output_config[i]["attn_ratio"] = layer_ratio[j]
                        logger.debug(f"Layer {i} {class_name} ratio: {layer_ratio[j]:.2f}")
                        j += 1
                    elif class_name in [model_helper.mamba_class_name, model_helper.mamba_uniql_class_name]:
                        output_config[i]["mamba_ratio"] = layer_ratio[j]
                        logger.debug(f"Layer {i} {class_name} ratio: {layer_ratio[j]:.2f}")
                        j += 1
                    else:
                        pass

        # Output dir
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_file_dir, "outputs", model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Store the layer-wise compression ratios
        filename = "uniform" if args.uniform else "layerwise_eps-" + str(args.epsilon)
        filename = filename + f"_ratio-{ratio}"
        if args.fine_grained:
            filename = filename + "_fine_grained"
        filename = filename + ".json"
        logger.info(f"Save the layer-wise configurations to {output_dir}/{filename}")
        with open(f"{output_dir}/{filename}", "w") as f:
            json.dump(output_config, f)

if __name__ == "__main__":

    set_deterministic(1234)
    parser = argparse.ArgumentParser(description="Get the layer-wise compression ratios for each layer by layer importance")
    parser = argparse_shared_options(parser)
    parser = argparse_calibration_options(parser)
    parser = argparse_compress_options(parser)
    parser.add_argument(
        '--cached_layer_importance', type=str, default=None,
        help='The json file to store/load the layer importance for the compressing the model (default: None)'
    )
    parser.add_argument(
        '--uniform', action='store_true', default=False,
        help='If set, use a uniform low-rank compression ratio for all linear layers. (default: False)'
    )
    parser.add_argument(
        '--fine_grained', action='store_true', default=False,
        help='If set, calculate the layer-wise compression ratio for each operator, e.g., MLP, Attention and Mamba, in a layer. (default: False)'
    )
    parser.add_argument(
        '--epsilon', type=float, default=0.1,
        help='Epsilon for entropic regularization when allocating layer-wise low-rank ratio. (default: 0.1)'
    )
    parser.add_argument(
        '--ratio_list', type=float, nargs='+', default=[0.75],
        help='A list of preserved low-rank ratios for linear layers (e.g., 0.5 0.75 1.0). We generate all configurations at once.'
    )
    args = parser.parse_args()
    if args.verbose:
        set_logger(logger, logging.DEBUG)
    else:
        set_logger(logger, logging.INFO)
    # run
    main(args)