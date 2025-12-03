import math
import fast_hadamard_transform_cuda

def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def next_power_of_two(x):
    """Return the smallest power of two >= x."""
    return 1 if x == 0 else 2**(x-1).bit_length()

def prev_power_of_two(x):
    """Return the largest power of two <= x."""
    if x < 1:
        # raise ValueError("Input must be >= 1.")
        return 0
    return 2**(x.bit_length() - 1)

def get_had_fn(dim):
    had_scale = 1.0 / math.sqrt(dim) # hadamard transform scaling factor
    if dim % 40 == 0 and is_pow2(dim // 40):
        N = 40
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_40N
    elif dim % 28 == 0 and is_pow2(dim // 28):
        N = 28
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_28N
    elif dim % 20 == 0 and is_pow2(dim // 20):
        N = 20
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_20N
    elif dim % 12 == 0 and is_pow2(dim // 12):
        N = 12
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform_12N
    else:
        assert (is_pow2(dim)), f"Invalid dim: {dim}. dim must be a power of 2."
        N = 2
        transform_fn = fast_hadamard_transform_cuda.fast_hadamard_transform
    return transform_fn, N, had_scale


def had_transform(w):
    saved_shape = w.shape
    dim = saved_shape[-1]
    # fast_hadamard_transform_cuda will create a new output tensor
    # see https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/fast_hadamard_transform.cpp#L140
    transform_fn, N, had_scale = get_had_fn(dim)
    w_H = transform_fn(w.reshape(-1, dim), had_scale)
    if w_H.shape != saved_shape:
        w_H = w_H.reshape(saved_shape)
    return w_H

