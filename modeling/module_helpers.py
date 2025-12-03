import logging
import torch
import torch.nn as nn

def get_size(module, return_storage=True):
    if module is None:
        return 0
    if isinstance(module, torch.nn.Parameter) or isinstance(module, torch.Tensor):
        return module.nelement() * module.element_size()
    param_size = 0
    buffer_size = 0
    for param in module.parameters():
        if return_storage:
            param_size += param.nelement() * param.element_size()
        else:
            param_size += param.nelement()
    for buffer in module.buffers():
        if return_storage:
            buffer_size += buffer.nelement() * buffer.element_size()
        else:
            buffer_size += buffer.nelement()
    return param_size + buffer_size



def split_linear(linear, split_dims=[]):

    in_features = linear.in_features
    weight = linear.weight.data # [Dout, Din]
    bias = linear.bias.data if linear.bias is not None else None

    d_start = 0
    linear_list = nn.ModuleList()
    for dim in split_dims:
        if dim <= 0:
            logging.debug(f"Skip splitting linear layer with dim {dim} in {linear.__class__.__name__}, split dims: {split_dims}")
            continue
        l = nn.Linear(in_features, dim, bias=bias is not None)
        d_end = d_start + dim
        l.weight.data = weight[d_start:d_end, :]
        if bias is not None:
            l.bias.data = bias[d_start:d_end]
        linear_list.append(l)
        d_start = d_end
    
    return linear_list

def merge_linear(linear_list):
    weight = torch.cat([linear.weight.data for linear in linear_list], dim=0) # [Dout, Din]
    bias_list = []
    for linear in linear_list:
        if linear.bias is not None:
            bias_list.append(linear.bias.data)
    bias = torch.cat(bias_list, dim=0) if len(bias_list) > 0 else None
    assert weight.shape[1] == bias.shape[0] if bias is not None else True, f"Weight and bias shape mismatch: {weight.shape[1]} != {bias.shape[0]}"
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    linear.weight.data = weight
    if bias is not None:
        linear.bias.data = bias
    return linear