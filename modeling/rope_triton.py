"""
This code is modified from https://github.com/linkedin/Liger-Kernel/blob/454e3d2e3492aec1ae4236be7e6aec156f6bced8/src/liger_kernel/ops/rope.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope_indices(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    indices_ptr,
    indices_row_stride,
    sl,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    cos_dim: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    pad_cos_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # q size: (bsz, seq_len, num_q_heads, head_dim)
    # q stride: (seq_len * num_q_heads * head_dim, num_q_heads * head_dim, head_dim, 1)
    # k size: (bsz, seq_len, num_kv_heads, head_dim)
    # k stride: (seq_len * num_kv_heads * head_dim, num_kv_heads * head_dim, head_dim, 1)

    # cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
    # stride: (seq_len * head_dim, head_dim, 1)
    pid = tl.program_id(0).to(tl.int64)

    # locate start address
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################

    # 1. program instances are laid out in a 1D vector of size bsz * seq_len, which
    # effectively represents a 2D grid of size [bsz, seq_len] with seq_len dimension
    # being the fastest changing dimension. Thus we can simply do pid // sl to get the batch index
    # and pid % sl to get the sequence index.
    # 2. We only need the left half of cos and sin matrix because the right half is just
    # a clone of the left half.
    batch_idx = pid // sl
    cos_row_idx = pid % sl
    cos = cos + tl.where(
        cos_bs == 1,
        cos_row_idx * cos_row_stride,
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride,
    )
    sin = sin + tl.where(
        cos_bs == 1,
        cos_row_idx * sin_row_stride,
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride,
    )

    cos_offsets = tl.arange(0, pad_cos_dim // 2)
    cos_mask = cos_offsets < cos_dim // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0).expand_dims(0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0).expand_dims(0)


    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
    # load indices
    repeat = n_qh // n_kh
    repeated_row_index = tl.arange(0, pad_n_qh) // repeat  # GQA
    q_indices_offsets = repeated_row_index[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    q_indices = tl.load(indices_ptr + q_indices_offsets, mask=first_q_mask, other=None) # indices.shape = (n_qh, hd)
    k_indices = tl.load(indices_ptr + first_half_k_offsets, mask=first_k_mask, other=None) # indices.shape = (n_kh, hd)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # q
    cos_row_broadcast_q = tl.broadcast_to(cos_row, (pad_n_qh, pad_cos_dim // 2))
    sin_row_broadcast_q = tl.broadcast_to(sin_row, (pad_n_qh, pad_cos_dim // 2))
    cos_row_gathered_q = tl.gather(cos_row_broadcast_q, q_indices, axis=1)
    sin_row_gathered_q = tl.gather(sin_row_broadcast_q, q_indices, axis=1)
    new_q_tile_1 = q_tile_1 * cos_row_gathered_q - q_tile_2 * sin_row_gathered_q
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row_gathered_q + q_tile_1 * sin_row_gathered_q
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    # k
    cos_row_broadcast_k = tl.broadcast_to(cos_row, (pad_n_kh, pad_cos_dim // 2))
    sin_row_broadcast_k = tl.broadcast_to(sin_row, (pad_n_kh, pad_cos_dim // 2))
    cos_row_gathered_k = tl.gather(cos_row_broadcast_k, k_indices, axis=1)
    sin_row_gathered_k = tl.gather(sin_row_broadcast_k, k_indices, axis=1)
    new_k_tile_1 = k_tile_1 * cos_row_gathered_k - k_tile_2 * sin_row_gathered_k
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row_gathered_k + k_tile_1 * sin_row_gathered_k
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def apply_rotary_pos_emb_indices_triton(q, k, cos, sin, indices):
    # transpose it back to the physical shape because Triton looks at the physical storage
    # note: q and k are incontiguous before the transformation and will become contiguous after transpose
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    assert indices.shape[0] == n_kv_head
    assert indices.shape[1] == head_dim

    n_row = batch_size * seq_len

    # ensure tensors passed into the kernel are contiguous. It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    indices = indices.contiguous()
    cos_batch_size = cos.shape[0]
    cos_dim = cos.shape[-1]
    pad_cos_dim = triton.next_power_of_2(cos_dim)

    # !!! q and k will be modified in place !!!
    _triton_rope_indices[(n_row,)](
        q,
        q.stride(1),
        k,
        k.stride(1),
        cos,
        cos.stride(-2),
        sin,
        sin.stride(-2),
        indices,
        indices.stride(0),
        seq_len,
        batch_size,
        cos_batch_size,
        cos_dim,
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        pad_cos_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q.transpose(1, 2), k.transpose(1, 2)