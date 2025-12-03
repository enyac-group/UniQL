"""
This file is a modified version of the original file from the GPTQ repo.
https://github.com/IST-DASLab/gptq
"""
import math
import gc

import torch
import torch.nn as nn

from quantize.quant_linear_layers import HadLinear

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def get_per_channel_scale(w, num_bits=4):
    max_q_val = 2**num_bits - 1 # max_q_val=15 when num_bits=4
    # Compute scale for each output channel
    s = torch.max(torch.abs(w), 1, keepdim=True)[0] # w: [Dout, Din] 
    s *= 2 / max_q_val  # 2 => symmetric, 2 / 15 when num_bits=4  # s: [Dout, 1] 
    return s

@torch.no_grad()
def quant(w, s, num_bits=4):
    # HY: use the simple quant in qqq_quantize_weights
    max_q_val = 2**num_bits - 1
    half_q_val = (max_q_val + 1) // 2
    # w: [Dout, Din], s: [Dout, 1]
    q_w = torch.round(w / (s + 1e-6)).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
    q_w += half_q_val # [0, 16]
    q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
    # Compute ref (dequantized)
    w_ref = (q_w - half_q_val) * s
    return w_ref.to(w.dtype)

class GPTQ:
  
    def __init__(self, layer, dtype=None):

        if not isinstance(layer, (nn.Linear, HadLinear)):
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}.")
        
        self.layer = layer # weight: [Dout, Din]
        self.dev = self.layer.weight.device
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = self.layer.weight.dtype
        self.out_dim = layer.weight.data.shape[0] # out dim
        self.in_dim = layer.weight.data.shape[1] # in dim
        self.H = torch.zeros((self.in_dim, self.in_dim), device=self.dev, dtype=self.dtype)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, module, inp, out):

        if type(inp) is tuple:
            inp = inp[0]
        assert type(inp) is torch.Tensor

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(1) # [b, d] -> [b, 1, d]
        n = inp.shape[0] # [b, seqlen, d]
        inp = inp.to(self.dtype)
        adds = torch.matmul(inp.transpose(1, 2), inp) # [b, d, seqlen] @ [b, seqlen, d]
        adds_sum = torch.sum(adds, dim=0) / n # [B, d, d] -> [d, d]
        adds_sum *= (n / (self.nsamples + n))
        self.H *= (self.nsamples / (self.nsamples + n))
        self.H += adds_sum
        self.nsamples += n
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    
    def fasterquant(self, group_size=128, percdamp=.01, w_bits=4):
        bits = w_bits # 4-bit quantization
        W = self.layer.weight.data.clone().to(self.dtype) # [Dout, Din]
        device = W.device
        
        # preprocess H
        H = self.H.clone().to(torch.float32)
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.in_dim, device=self.dev)
        H[diag, diag] += damp 
        # cholesky must be torch.float32
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True).to(self.dtype)
        Hinv = H
        
        # init Losses and Q
        assert group_size <= self.in_dim
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        n_groups = math.ceil(self.in_dim / group_size)
        group_scale = torch.zeros(n_groups, self.out_dim, dtype=torch.float32, device=device)   # QQQ requires [n_groups, out_dim]
        for i1 in range(0, self.in_dim, group_size):
            gidx = i1 // group_size
            i2 = min(i1 + group_size, self.in_dim)
            count = i2 - i1
            # get weight group
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            # Get per-group scale and zero
            per_group_scale = get_per_channel_scale(W1, num_bits=bits)
            group_scale[gidx] = per_group_scale.squeeze()
            for i in range(count):
                w = W1[:, i].clone() # [Dout]
                d = Hinv1[i, i]
                q = quant(w.unsqueeze(1), per_group_scale, num_bits=bits).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        torch.cuda.synchronize()
        
        Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype) # fake quantized weight
        self.layer.weight.data = Q.contiguous()
        self.layer.apply_gptq = True
        self.layer.bits = bits
        self.layer.group_size = group_size
        self.layer.group_scale = group_scale    # QQQ requires [n_groups, out_dim]
        assert not torch.isnan(Q).any(), f"Quantized weight has nan"

        del Losses
        del H
        del W
        torch.cuda.empty_cache()

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
        gc.collect()