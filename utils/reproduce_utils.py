import random
import numpy as np
import torch

def set_deterministic(seed):
    # Fix all possible random seef for reproduce
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)