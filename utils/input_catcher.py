import torch
import torch.nn as nn

# we use a Catcher to collect the inputs for the first layer
class CatcherExit(Exception):
    pass

# Setup caching mechanism
class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.i = 0
        self.module = module
        # Copy all attributes from the original module
        for name, attr in module.__dict__.items():
            if not name.startswith('_'):  # Skip private attributes
                setattr(self, name, attr)
    
    def init_input_buffer(self, nsamples, seqlen, dim, **kwargs):
        self.input_buffer = torch.empty((nsamples, seqlen, dim), **kwargs)
        self.cached_kwargs = []

    def forward(self, inp, **kwargs):
        self.input_buffer[self.i] = inp
        self.i += 1
        if "position_embeddings" in kwargs:
            kwargs["position_embeddings"] = (
                kwargs["position_embeddings"][0].to(self.input_buffer.dtype),
                kwargs["position_embeddings"][1].to(self.input_buffer.dtype))
        self.cached_kwargs.append(kwargs)
        raise CatcherExit