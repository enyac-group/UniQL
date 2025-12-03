import torch

from quantize.hadamard_helpers import get_had_fn

class Hadamard(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform_fn, self.N, self.had_scale = get_had_fn(dim)

    def to(self, *args, **kwargs):
        super(Hadamard, self).to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.transform_fn(x.contiguous(), self.had_scale) 

    def __repr__(self):
        return f"Hadamard(dim={self.dim})"
