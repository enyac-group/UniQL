"""
This code is modified from:
https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
"""
import numpy as np

import torch

MARLIN_QQQ_TILE = 16
MARLIN_QQQ_MIN_THREAD_N = 64
MARLIN_QQQ_MIN_THREAD_K = 128
MARLIN_QQQ_MAX_PARALLEL = 16
GPTQ_MARLIN_TILE = 16

SUPPORTED_NUM_BITS = [4]
SUPPORTED_GROUP_SIZES = [-1, 128]

class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


@torch.no_grad()
def permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    # assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_n = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


@torch.no_grad()
def get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single



@torch.no_grad()
def get_w4a16_permute_weights(q_w, size_k, size_n, num_bits, perm, group_size):
    # Permute
    q_w = permute_weights(q_w, size_k, size_n, perm) # [k, n] -> [-1, 1024]

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device) # [-1, 1024] -> [-1, 1024/pack_factor]

    return q_packed


@torch.no_grad()
def get_fake_quantize_weights(w: torch.Tensor, num_bits: int, group_size: int = -1):
    orig_device = w.device
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if group_size < size_k:

        padding_k = (group_size - (size_k % group_size)) % group_size  # avoid 128 padding if already divisible
        if padding_k > 0:
            w = torch.nn.functional.pad(w, (0, 0, 0, padding_k), mode='constant', value=0)  # pad rows (dim 0) with zeros at the bottom
        # Reshape to [groupsize, -1]
        w = w.reshape((-1, group_size, size_n)) # [Din, Dout] -> [ngroup, group_size, size_n]

        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        # Compute scale for each group
        scale = torch.max(torch.abs(w), 1, keepdim=True)[0]
        scale *= 2 / max_q_val  # 2 => symmetric

        # Quantize
        q_w = torch.round(w / scale).int() # round([-7.5, 7.5]) -> [-8, 8], .int() will replace NaN with 0
        q_w += half_q_val # [0, 16]
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        # Compute ref (dequantized)
        w_ref = (q_w - half_q_val) * scale

        q_w = q_w.reshape((-1, size_n)).contiguous()
        w_ref = w_ref.reshape((-1, size_n)).contiguous()
        scale = scale.reshape((-1, size_n)).contiguous()
    else:
        assert group_size == size_k
        max_q_val = 2**(num_bits - 1) - 1

        # Compute scale for each channel
        scale = torch.max(torch.abs(w), 0, keepdim=True)[0]
        scale /= max_q_val

        # Quantize
        q_w = torch.round(w / scale).int()
        q_w = torch.clamp(q_w, -max_q_val, max_q_val)
        # Compute ref (dequantized)
        w_ref = q_w * scale

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        scale.to(device=orig_device),
    )


@torch.no_grad()
def get_w4a16_weight_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


@torch.no_grad()
def w4a16_permute_scales(scale, size_k, size_n, group_size):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        assert group_size == 128, "Only group_size 128 is supported for per-group quantization"
        scale = scale.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        assert group_size == size_k
        scale = scale.reshape(
            (-1, len(scale_perm_single)))[:, scale_perm_single]
    scale = scale.reshape((-1, size_n)).contiguous()

    return scale


@torch.no_grad()
def w4a16_quantize(
    w: torch.Tensor, num_bits: int,
    group_size: int = -1, scale: torch.Tensor = None,
    pad_out: int = 0
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    quant_type = "per-channel" if group_size == size_k else "per-group"

    assert num_bits in SUPPORTED_NUM_BITS, f"Unsupported num_bits = {num_bits}"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    # Quantize
    if scale is None:
        w_ref, q_w, scale = get_fake_quantize_weights(w, num_bits, group_size) # per-group
    else: # scale
        # w is already (fake) quantized, just represent w in integer here
        max_q_val = 2**num_bits - 1
        half_q_val = (max_q_val + 1) // 2

        padding_k = (group_size - (size_k % group_size)) % group_size  # avoid 128 padding if already divisible
        if padding_k > 0:
            w = torch.nn.functional.pad(w, (0, 0, 0, padding_k), "constant", 0)  # pad rows (dim 0) with zeros at the bottom
        
        w_ref = w.clone() # k (Din) is padded here, so we can get new size_k
        w = w.reshape((-1, group_size, size_n))
        q_w = ((w / scale.unsqueeze(1)) + half_q_val).round().int()
        q_w = torch.clamp(q_w, 0, max_q_val) #[0, 15]
        q_w = q_w.reshape((size_k + padding_k, size_n)).contiguous()

    if pad_out != 0:    
        w_ref = torch.nn.functional.pad(w_ref, (0, pad_out, 0, 0), "constant", 0) # w_ref: [Din, Dout]
        q_w = torch.nn.functional.pad(q_w, (0, pad_out, 0, 0), "constant", 0) # q_w: [Din, Dout]
        scale = torch.nn.functional.pad(scale, (0, pad_out), "constant", 0) # scale: [n_group, Dout]
    
    size_k, size_n = w_ref.shape # new size after padding k (Din) and n (Dout)

    # weight permutation
    weight_perm = get_w4a16_weight_perms()
    marlin_qqq_q_w = get_w4a16_permute_weights(q_w, size_k, size_n, num_bits,
                                        weight_perm, group_size)
    marlin_scale = w4a16_permute_scales(
        scale, size_k, size_n, group_size)

    # Create result
    res_list = [
        w_ref, marlin_qqq_q_w, marlin_scale
    ]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


@torch.no_grad()
def w4a16_unpermute_scales(permuted_scale, size_k, size_n, group_size):
    """
    Recover original (pre-permuted) scale from the permuted version used for w4a16 GeMM.

    Args:
        permuted_scale (Tensor): shape [size_k // group_size, size_n]
        size_k (int)
        size_n (int)
        group_size (int)

    Returns:
        Tensor: original scale shape [num_groups, num_scales_per_group]
    """
    scale_perm_group, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        assert group_size == 128, "Only group_size 128 is supported for per-group quantization"
        scale_perm = scale_perm_group
    else:
        assert group_size == size_k
        scale_perm = scale_perm_single
    num_scales_per_group = len(scale_perm)

    # Compute reverse permutation
    scale_perm_reverse = np.argsort(scale_perm)

    # Unpermute
    scale_orig = permuted_scale.reshape(-1, num_scales_per_group)
    scale_orig = scale_orig[:, scale_perm_reverse]
    scale_orig = scale_orig.reshape(-1, size_n).contiguous()

    return scale_orig


@torch.no_grad()
def unpermute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    """
    Inverse of permute_weights.
    Input q_w is [size_k // tile, size_n * tile] with perm applied.
    Output shape: [size_k, size_n]
    """
    q_w = q_w.clone()  # avoid modifying in-place
    q_w = q_w.reshape((-1, perm.numel()))
    
    # Compute inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)

    # Apply inverse perm
    q_w = q_w[:, inv_perm].reshape((size_k // tile, size_n * tile))

    # Invert tiling: reshape and permute back
    q_w = q_w.reshape((size_k // tile, size_n // tile, tile, tile))
    q_w = q_w.permute((0, 2, 1, 3)).contiguous()
    q_w = q_w.reshape(size_k, size_n)

    return q_w


@torch.no_grad()
def unpack_w4a16_permute_weights(q_packed, size_k, size_n, num_bits, perm, group_size):
    """
    Unpacks int32 packed q_packed into 4-bit q_w and inverts permutation.
    Output: q_w [size_k, size_n] in int32
    """
    pack_factor = get_pack_factor(num_bits)  # e.g., 8 for 4-bit
    orig_device = q_packed.device

    q_packed = q_packed.cpu().numpy().astype(np.uint32)
    num_rows, packed_cols = q_packed.shape
    unpacked_cols = packed_cols * pack_factor

    q_w = np.zeros((num_rows, unpacked_cols), dtype=np.uint32)

    for i in range(pack_factor):
        shift = num_bits * i
        mask = (1 << num_bits) - 1
        q_w[:, i::pack_factor] = (q_packed >> shift) & mask

    q_w = torch.from_numpy(q_w.astype(np.int32)).to(orig_device)

    # Unpermute
    q_w = unpermute_weights(q_w, size_k, size_n, perm)

    return q_w
