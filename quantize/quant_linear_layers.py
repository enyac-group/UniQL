import torch
import torch.nn as nn
import torch.nn.functional as F

import quant_linear_cuda

from quantize.hadamard_helpers import get_had_fn
from quantize.quantize_helpers import w4a16_quantize
from quantize.quantize_helpers import MARLIN_QQQ_MIN_THREAD_N, MARLIN_QQQ_MAX_PARALLEL

class W4A16B16O16Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=False, group_size=128, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.size_n, self.size_k = out_features, in_features
        
        self.pad_out = 0
        if self.size_n % 256 != 0:
            self.pad_out = 256 - self.size_n % 256
        self.size_n = self.size_n + self.pad_out

        self.pad_in = 0
        if self.size_k % group_size != 0:
            self.pad_in = group_size - self.size_k % group_size
        self.size_k = self.size_k + self.pad_in

        self.max_par = 16
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        max_workspace_size = ((self.size_n // MARLIN_QQQ_MIN_THREAD_N) * MARLIN_QQQ_MAX_PARALLEL)
        self.register_buffer('workspace', torch.zeros(
            max_workspace_size, dtype=torch.int32, **factory_kwargs))
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer('weight', torch.empty(
            (self.size_k//16, self.size_n*16 // 8),
            dtype=torch.int32, **factory_kwargs))
        self.register_buffer('scale', torch.empty(
            (self.size_k//group_size, self.size_n),
            dtype=torch.float16, **factory_kwargs))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.empty(
                out_features, dtype=torch.float16, **factory_kwargs)))
        else:
            self.bias = None

    @classmethod
    def from_fp16(cls, originalLayer: nn.Linear):
        # assert originalLayer.bias is None, "Not support bias yet"
        # The linear kernel only supports symmetric quantization, so we only have scales
        bits = 4
        group_size = 128 if originalLayer.in_features > 128 else -1
        group_scale = None
        if hasattr(originalLayer, "apply_gptq") and originalLayer.apply_gptq == True:
            bits = originalLayer.bits
            group_size = originalLayer.group_size
            group_scale = originalLayer.group_scale # [n_groups, out_dim]
            # Marlin requires float16 scaling factors
            group_scale = group_scale.to(torch.float16)
        assert bits == 4, "Only support 4-bit quantization"
        if group_size not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        
        device = originalLayer.weight.device
        qlinear = cls(originalLayer.in_features, originalLayer.out_features, device=device)
        qlinear.pad_out = 0
        W = originalLayer.weight # [Dout, Din]
        qlinear.size_n, qlinear.size_k = W.shape
        if W.shape[0] % 256 != 0:
            qlinear.pad_out = 256 - W.shape[0] % 256

        # W16 per-channel per-group quantization to W4
        W_t = W.cpu().to(torch.float16).t().contiguous() # [Dout, Din] -> [Din, Dout], move to CPU to save memory
        group_scale = group_scale.cpu() if group_scale is not None else None
        w_ref, q_w, scale = w4a16_quantize(
            W_t, bits, group_size, group_scale, pad_out=qlinear.pad_out)
        qlinear.size_k, qlinear.size_n = w_ref.shape # the size_k and size_n are padded here, see w4a16_quantize function
        qlinear.pad_in = w_ref.shape[0] - W_t.shape[0] # [Din, Dout]
        qlinear.max_par = 16
        qlinear.weight = q_w.to(device)
        qlinear.scale = scale.to(device) # weight scale

        if originalLayer.bias is not None:
            qlinear.bias = nn.Parameter(originalLayer.bias.data.clone())
        
        return qlinear

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        # this contiguous is necessary for batch size > 1 for lm_head
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L281
        x = x.view(-1, x_shape[-1]).contiguous() # must squeeze the tensor first
        if self.pad_in != 0:
            x = torch.nn.functional.pad(x, (0, self.pad_in), "constant", 0)
        y = quant_linear_cuda.w4a16o16_gemm(
            x,
            self.weight,
            self.scale,
            self.workspace,
            x.shape[0],     # m: batch size * seq_len
            self.size_n,    # n: out_features
            self.size_k,    # k: in_features
            False, -1, -1, -1, self.max_par
        )
        if self.pad_out != 0:
            y = y[:, 0:-self.pad_out]
        y = y.view(*x_shape[:-1], -1)

        if self.bias is not None:
            y = y + self.bias
        
        return y

    def __repr__(self):
        return f"W4A16B16O16Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class HadLinear(torch.nn.Linear):

    def __init__(self,
        originalLayer: nn.Linear,
        input_transform=True,
        output_transform=False,
        fuse_had=False
    ):
        assert originalLayer.weight.is_cuda, "Hadamard transform must be on CUDA"
        super().__init__(
            originalLayer.in_features,
            originalLayer.out_features,
            True if originalLayer.bias is not None else False,
            originalLayer.weight.device,
            originalLayer.weight.dtype,
        )
        
        # Do not fuse the Hadamard matrix here, so we can do weight re-ordering
        self.input_transform = input_transform
        if input_transform:
            self.input_transform_fn, self.Nin, self.had_scale_in = get_had_fn(
                originalLayer.in_features)
        
        self.output_transform = output_transform
        if output_transform:
            self.output_transform_fn, self.Nout, self.had_scale_out = get_had_fn(
                originalLayer.out_features)
            
        self.weight.data = originalLayer.weight.data
        if originalLayer.bias is not None:
            assert self.output_transform is False, "Bias is not supported for output_transform"
            self.bias.data = originalLayer.bias.data
        
        self.fuse_had = fuse_had
        if self.fuse_had:
            self.fuse_hadamard()

    def fuse_hadamard(self):
        W = self.weight.data.clone()
        if self.input_transform:
            W = self.input_transform_fn(W, self.had_scale_in)
        if self.output_transform:
            W_t = self.output_transform_fn(W.t(), self.had_scale_out)
            W = W_t.t()
        self.weight.data = W.contiguous()
        self.fuse_had = True
    
    def forward(self, x):
        w_H = self.weight.clone()
        if not self.fuse_had and self.input_transform:
            w_H = self.input_transform_fn(w_H, self.had_scale_in)
        if not self.fuse_had and self.output_transform:
            w_H_t = self.output_transform_fn(w_H.t(), self.had_scale_out)
            w_H = w_H_t.t()
        return F.linear(x, w_H, self.bias)

    def __repr__(self):
        return f"HadLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}" + \
                    f"input_transform={self.input_transform}, output_transform={self.output_transform}" + \
                    f"fuse_had={self.fuse_had})"

