import os
import logging

import torch
from einops import rearrange

from utils.logger_utils import set_logger

logger = logging.getLogger(os.path.basename(__file__))
set_logger(logger, logging.INFO)


@torch.no_grad()
def mlp_compression(up_proj, down_proj, gate_proj, down_x, rank, ridge_lambda=1e-4):
    """
    Implements Algorithm 1: Type-I compression for MLP by Nyström approximation.

    Args:
        up_proj: nn.Linear, the up projection layer
        down_proj: nn.Linear, the down projection layer
        gate_proj: nn.Linear, the gate projection layer
        down_x: [B, T, d_h] input hidden states to up_proj and gate_proj
        rank: int, target compressed rank
        ridge_lambda: float, regularization strength λ for ridge leverage

    Returns:
        (WU_trunc, WD_trunc): compressed weights
    """
    device = up_proj.weight.device
    saved_dtype = up_proj.weight.dtype

    n = down_x.shape[0] # [b, l, d_int]
    X = down_x # [b, l, h] we use fp16 here, and convert to fp32 below batched by batched to save memory
    C_sigma = 0
    for i in range(n):
        Xi = X[i].to(torch.float32) # for numerical stability
        XiT = Xi.t()
        C_sigma += torch.matmul(XiT, Xi) / n # we must use float32 for the numerical stability
    del X, Xi, XiT
    torch.cuda.empty_cache()

    # Ridge leverage score: diag(C (C + λI)^(-1))
    I = torch.eye(C_sigma.shape[0], device=C_sigma.device, dtype=C_sigma.dtype)
    C_ridge_inv = torch.linalg.inv(C_sigma + ridge_lambda * I)
    scores = torch.diag(C_sigma @ C_ridge_inv)  # [d_int]
    del C_ridge_inv, I
    torch.cuda.empty_cache()

    # Top-k selection
    k = rank
    topk = torch.topk(scores, k=k, largest=True).indices
    Sk = torch.eye(C_sigma.shape[0], device=device)[:, topk]  # [d_int, r]

    # Reconstruct compressed WU and WD
    Wu = up_proj.weight.data.to(torch.float32).t() # [d_int, h] -> [h, d_int]
    Wu_trunc =  Wu @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
    Wu_trunc = Wu_trunc.contiguous().to(saved_dtype).t() # [h, r] -> [r, h]
    if gate_proj is not None:
        Wg = gate_proj.weight.data.to(torch.float32).t() # [d_int, h] -> [h, d_int]
        Wg_trunc = Wg @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
        Wg_trunc = Wg_trunc.contiguous().to(saved_dtype).t()
    else:
        Wg_trunc = None
    Wd = down_proj.weight.data.to(torch.float32).t() # [h, d_int] -> [d_int, h]
    # inv([r, d_int] @ [d_int, d_int] @ [d_int, r]) @ [r, d_int] @ [d_int, d_int] @ [d_int, h]
    Wd_trunc = Sk.T @ Wd
    Wd_trunc = Wd_trunc.contiguous().to(saved_dtype).t()

    up_proj_trunc = torch.nn.Linear(Wu_trunc.shape[1], Wu_trunc.shape[0], bias=up_proj.bias is not None, dtype=saved_dtype, device=device)
    up_proj_trunc.weight.data = Wu_trunc
    if gate_proj is not None:
        gate_proj_trunc = torch.nn.Linear(Wg_trunc.shape[1], Wg_trunc.shape[0], bias=gate_proj.bias is not None, dtype=saved_dtype, device=device)
        gate_proj_trunc.weight.data = Wg_trunc
    else:
        gate_proj_trunc = None 
    down_proj_trunc = torch.nn.Linear(Wd_trunc.shape[1], Wd_trunc.shape[0], bias=down_proj.bias is not None, dtype=saved_dtype, device=device)
    down_proj_trunc.weight.data = Wd_trunc
    return up_proj_trunc, down_proj_trunc, gate_proj_trunc


@torch.no_grad()
def attn_vo_compression(v_proj, o_proj, inp, rank, num_attention_heads, num_key_value_heads):
    """
    Args:
        v_proj: nn.Linear, the value projection layer
        o_proj: nn.Linear, the output projection layer
        inp: [B, T, d_h] input hidden states to v_proj and o_proj
        rank: int, target compressed rank
        num_attention_heads: int, number of attention heads
        num_key_value_heads: int, number of key value heads

    Returns:
        (new_v_proj_weight, new_o_proj_weight): compressed weight matrices
    """
    assert num_attention_heads == num_key_value_heads, "num_attention_heads must be equal to num_key_value_heads"
    num_heads = num_key_value_heads
    saved_dtype = v_proj.weight.dtype
    device = v_proj.weight.device
    # Reshape original weights per head
    # v_proj.weight [num_heads * head_dim, hidden_size]
    # o_proj.weight [hidden_size, num_heads * head_dim]
    W_V = rearrange(v_proj.weight, '(n d) h -> n h d', n=num_heads).to(torch.float32)  # [n, h, d]
    W_O = rearrange(o_proj.weight, 'h (n d) -> n d h', n=num_heads).to(torch.float32)  # [n, d, h]

    # Compute input correlation C = sum XᵀX one by one to save memory
    inp = inp.to(torch.float32)  # [b, l, h]
    C = sum(torch.matmul(inp[i].T, inp[i]) / inp.shape[0] for i in range(inp.shape[0]))

    # C^{1/2}
    eigvals, eigvecs = torch.linalg.eigh(C)  # For symmetric matrices
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
    C_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    C_inv_sqrt = eigvecs @ torch.diag(1.0 / sqrt_eigvals) @ eigvecs.T  # Inverse square root
    
    WV_trunc_list = []
    WO_trunc_list = []

    for j in range(num_heads):
        WV_j = W_V[j]  # [h, d]
        WO_j = W_O[j]  # [d, h]

        # Step 1: SVD of C^{1/2} @ WV_j
        A = C_sqrt @ WV_j  # [h h] @ [h, d]
        U, S, Vh = torch.linalg.svd(A.to(torch.float64), full_matrices=False) # [h, d], [d], [d, d]
        U, S, Vh = U.to(torch.float32), S.to(torch.float32), Vh.to(torch.float32)

        # Step 2: SVD of ΣVᵀ @ WO_j
        AV = torch.diag(S) @ Vh @ WO_j  # [d, d] @ [d, d] @ [d, h]
        U_prime, S_prime, Vh_prime = torch.linalg.svd(AV.to(torch.float64), full_matrices=False) # [d, d], [d], [d, h]
        U_prime, S_prime, Vh_prime = U_prime.to(torch.float32), S_prime.to(torch.float32), Vh_prime.to(torch.float32)

        # Step 3: Construct compressed WV_j and WO_j, Apply quantization-aware decomposition
        WV_trunc_j = C_inv_sqrt @ U @ U_prime[:, :rank] @ torch.diag(S_prime[:rank])  # [h, h] @ [h, d] @ [d, r]
        WO_trunc_j = Vh_prime[:rank, :]  # [r, r] @ [r, h]

        WV_trunc_list.append(WV_trunc_j.contiguous().to(saved_dtype))  # [h, r]
        WO_trunc_list.append(WO_trunc_j.contiguous().to(saved_dtype))  # [r, h]

    # Stack all heads
    WV_trunc = torch.stack(WV_trunc_list, dim=0)  # [n, h, r]
    WO_trunc = torch.stack(WO_trunc_list, dim=0)  # [n, r, h]
    WV_trunc = rearrange(WV_trunc, 'n h r -> (n r) h')
    WO_trunc = rearrange(WO_trunc, 'n r h -> h (n r)')

    v_proj_trunc = torch.nn.Linear(WV_trunc.shape[1], WV_trunc.shape[0], bias=v_proj.bias is not None, dtype=saved_dtype, device=device)
    v_proj_trunc.weight.data = WV_trunc
    o_proj_trunc = torch.nn.Linear(WO_trunc.shape[1], WO_trunc.shape[0], bias=o_proj.bias is not None, dtype=saved_dtype, device=device)
    o_proj_trunc.weight.data = WO_trunc
    del C, C_sqrt, C_inv_sqrt
    return v_proj_trunc, o_proj_trunc


@torch.no_grad()
def attn_qk_compression(q_proj, k_proj, inp, rank, num_attention_heads, num_key_value_heads, position_embeddings=None, rotary_fn=None, rotate_half=False):
    """
    Args:
        q_proj: nn.Linear for query projection
        k_proj: nn.Linear for key projection
        inp: [B, T, d_h] input tensor to the attention layer
        rank: int, target compressed rank
        num_attention_heads: int, number of attention heads
        num_key_value_heads: int, number of key value heads
        position_embeddings: [T, d_h] position embeddings
        rotary_fn: function to apply rotary position embedding
        rotate_half: bool, whether to rotate half of the hidden states

    Returns:
        (q_proj_trunc, k_proj_trunc, indices): new compressed weight matrices [num_heads * k, d] and indices
    """
    assert num_attention_heads == num_key_value_heads, "num_attention_heads must be equal to num_key_value_heads"
    num_heads = num_key_value_heads
    saved_dtype = q_proj.weight.dtype
    device = q_proj.weight.device
    # Reshape weights per head
    # q_proj.weight [num_heads * head_dim, hidden_size]
    # k_proj.weight [num_heads * head_dim, hidden_size]
    WQ = rearrange(q_proj.weight, '(n d) h -> n h d', n=num_heads).to(torch.float32)  # [n, h, d]
    WK = rearrange(k_proj.weight, '(n d) h -> n h d', n=num_heads).to(torch.float32)  # [n, h, d]

    # Project inputs
    X = inp.to(torch.float32)  # [b, l, h]

    WQ_trunc_list = []
    WK_trunc_list = []
    indices_list = []

    for j in range(num_heads):
        WQ_j = WQ[j]  # [h, d]
        WK_j = WK[j]  # [h, d]

        Q_j = X @ WQ_j # [b, l, h] @ [h, d] -> [b, l, d]
        K_j = X @ WK_j # [b, l, h] @ [h, d] -> [b, l, d]
        # Get activations
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q_j, K_j = Q_j.unsqueeze(1), K_j.unsqueeze(1) # [b, l, d] -> [b, 1, l, d]
            Q_j, K_j = rotary_fn(Q_j, K_j, cos, sin)
            Q_j, K_j = Q_j.squeeze(1), K_j.squeeze(1)

        # Compute correlation matrices
        CQ_j = torch.matmul(Q_j.transpose(1, 2), Q_j) # [b, d, l] @ [b, l, d] -> [b, d, d]
        CQ_j = CQ_j.sum(dim=0) / CQ_j.shape[0]  # Average over batch, [d, d]
        CK_j = torch.matmul(K_j.transpose(1, 2), K_j) # [b, d, l] @ [b, l, d] -> [b, d, d]
        CK_j = CK_j.sum(dim=0) / CK_j.shape[0]  # Average over batch, [d, d]

        # Compute CQ^{1/2} and CK^{1/2}
        eigvals, eigvecs = torch.linalg.eigh(CQ_j.to(torch.float32))  # For symmetric matrices
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
        CQ_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

        eigvals, eigvecs = torch.linalg.eigh(CK_j.to(torch.float32))  # For symmetric matrices
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
        CK_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
        
        # Compute norm scores
        s = (CQ_sqrt.norm(dim=0) * CK_sqrt.norm(dim=0))  # [d]
        # rotate half: q = [q_rot, q_pass], k = [k_rot, k_pass]
        if rotate_half:
            quater = s.shape[0] // 4 # d // 4
            s_rot_pairs = s[:quater] + s[quater:2*quater]  # shape [d//4]
            topk_rot_pair_indices = torch.topk(s_rot_pairs, k=rank//4, largest=True).indices  # shape [rank // 4]
            topk_rot = torch.cat([topk_rot_pair_indices, topk_rot_pair_indices + quater], dim=0)  # shape [rank // 2]
            indices_list.append(topk_rot)
            # then we do the half that is not rotated
            half = s.shape[0] // 2 # d // 2
            topk_pass = half + torch.topk(s[:half], k=rank//2, largest=True).indices  # shape [rank // 2]
            # final concat the two halves
            topk = torch.cat([topk_rot, topk_pass], dim=0)  # shape [rank]
            S_k = torch.eye(s.shape[0], device=X.device, dtype=X.dtype)[:, topk]  # [d, r]
        else:
            d = s.shape[0]
            assert d % 2 == 0, "d must be even for rotation-based pairing"
            assert rank % 2 == 0, "rank must be even for rotation-based pairing"
            half = d // 2
            k_pairs = rank // 2
            # Compute pair scores
            s_pairs = s[:half] + s[half:]  # shape [d//2]
            topk_pair_indices = torch.topk(s_pairs, k=k_pairs, largest=True).indices  # shape [k_pairs]
            topk = torch.cat([topk_pair_indices, topk_pair_indices + half], dim=0)  # shape [rank]
            indices_list.append(topk)
            S_k = torch.eye(d, device=X.device, dtype=X.dtype)[:, topk]  # [d, r]

        # Apply compression
        WQ_j_trunc = WQ_j @ S_k # [h, d] @ [d, r]
        WK_j_trunc = WK_j @ S_k # [h, d] @ [d, r]

        WQ_trunc_list.append(WQ_j_trunc.to(saved_dtype)) # [h, r]
        WK_trunc_list.append(WK_j_trunc.to(saved_dtype)) # [h, r]

        del CQ_sqrt, CK_sqrt, S_k, Q_j, K_j, CQ_j, CK_j
        torch.cuda.empty_cache()  # Free memory after each head

    # Concatenate all heads, and rearrange
    WQ_trunc = torch.stack(WQ_trunc_list, dim=0) # [n, h, r]
    WK_trunc = torch.stack(WK_trunc_list, dim=0) # [n, h, r]
    WQ_trunc = rearrange(WQ_trunc, 'n h r -> (n r) h')
    q_proj_trunc = torch.nn.Linear(WQ_trunc.shape[1], WQ_trunc.shape[0], bias=q_proj.bias is not None, dtype=saved_dtype, device=device)
    q_proj_trunc.weight.data = WQ_trunc
    WK_trunc = rearrange(WK_trunc, 'n h r -> (n r) h')
    k_proj_trunc = torch.nn.Linear(WK_trunc.shape[1], WK_trunc.shape[0], bias=k_proj.bias is not None, dtype=saved_dtype, device=device)
    k_proj_trunc.weight.data = WK_trunc
    indices = torch.stack(indices_list, dim=0) # [n, r]
    return q_proj_trunc, k_proj_trunc, indices


@torch.no_grad()
def attn_vo_compression_gqa(v_proj, o_proj, inp, rank, num_attention_heads, num_key_value_heads):
    """
    Type-III Compression with GQA: Shared value projection per group.
    """
    group_size = num_attention_heads // num_key_value_heads
    saved_dtype = v_proj.weight.dtype
    device = v_proj.weight.device

    # Reshape weights
    WV = rearrange(v_proj.weight, '(g d) h -> g h d', g=num_key_value_heads).to(torch.float32)  # [g, h, d]
    WO = rearrange(o_proj.weight, 'h (n d) -> n d h', n=num_attention_heads).to(torch.float32)  # [n, d, h]

    bias_V = None
    if v_proj.bias is not None:
        bias_V = rearrange(v_proj.bias, '(g d) -> g d', g=num_key_value_heads).to(torch.float32)  # [g, d]
    bias_O = None
    if o_proj.bias is not None:
        bias_O = o_proj.bias.to(torch.float32)  # [h]

    # Compute input correlation C = sum XᵀX one by one to save memory
    inp = inp.to(torch.float32)  # [b, l, h]
    C = sum(torch.matmul(inp[i].T, inp[i]) / inp.shape[0] for i in range(inp.shape[0])) # [h, h]
    
    eigvals, eigvecs = torch.linalg.eigh(C)  # For symmetric matrices
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
    C_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
    C_inv_sqrt = eigvecs @ torch.diag(1.0 / sqrt_eigvals) @ eigvecs.T  # Inverse square root

    WV_trunc_list = []
    WO_trunc_list = []
    bias_list = [] # the fused VO bias

    for g in range(num_key_value_heads):
        # Shared W_V for group
        WV_group = WV[g]  # [h, d]
        # A = WV_group  # [h, h] @ [h, d] -> [h, d]
        A = C_sqrt @ WV_group  # [h, h] @ [h, d] -> [h, d]
        U, S, Vh = torch.linalg.svd(A, full_matrices=False) # [h, d], [d], [d, d]
        U_k = U[:, :rank]  # [h, r]
        S_k = S[:rank]
        V_k = Vh[:rank, :]  # [r, d]
        WV_trunc = (C_inv_sqrt @ U_k @ torch.diag(S_k)).to(saved_dtype)  # [h, h] @ [h, r] @ [r, r] -> [h, r]
        WV_trunc_list.append(WV_trunc)

        # Compress output projections for all heads in group
        for i in range(group_size):
            idx = g * group_size + i
            if bias_V is not None:
                fused_bias = bias_V[g] @ WO[idx] # [d] @ [d, h] -> [h]
                if bias_O is not None:
                    fused_bias = fused_bias + bias_O[idx] # TODO: bias_O should be replaced with closed-form solution
                bias_list.append(fused_bias)
            WO_trunc = (V_k @ WO[idx]).to(saved_dtype)  # [r, d] @ [d, h] -> [r, h]
            WO_trunc_list.append(WO_trunc)

    # Stack and reshape
    WV_trunc = torch.stack(WV_trunc_list, dim=0)  # [g, h, r]
    WO_trunc = torch.stack(WO_trunc_list, dim=0)  # [n, r, h]
    WV_trunc = rearrange(WV_trunc, 'g h r -> (g r) h')
    WO_trunc = rearrange(WO_trunc, 'n r h -> h (n r)')
    v_proj_trunc = torch.nn.Linear(WV_trunc.shape[1], WV_trunc.shape[0], bias=None, dtype=saved_dtype, device=device) # no bias for value projection
    v_proj_trunc.weight.data = WV_trunc
    o_proj_trunc = torch.nn.Linear(WO_trunc.shape[1], WO_trunc.shape[0], bias=bias_V is not None or bias_O is not None, dtype=saved_dtype, device=device) # fused bias for output projection
    o_proj_trunc.weight.data = WO_trunc
    if bias_V is not None or bias_O is not None:
        bias_O_trunc = sum(bias_list)  # [h], the fused bias for output projections
        o_proj_trunc.bias.data = bias_O_trunc.flatten().to(saved_dtype)
    return v_proj_trunc, o_proj_trunc


@torch.no_grad()
def attn_qk_compression_gqa(q_proj, k_proj, inp, rank, num_attention_heads, num_key_value_heads, position_embeddings=None, rotary_fn=None, rotate_half=False):
    """
    Args:
        q_proj: nn.Linear for query projection
        k_proj: nn.Linear for key projection
        inp: [B, T, d_h] input tensor to the attention layer
        rank: int, target compressed rank
        num_attention_heads: int, number of attention heads
        num_key_value_heads: int, number of key value heads
        position_embeddings: [T, d_h] position embeddings
        rotary_fn: function to apply rotary position embedding
        rotate_half: bool, whether to rotate half of the hidden states

    Returns:
        (q_proj_trunc, k_proj_trunc, indices): new compressed weight matrices [num_heads * k, d] and indices
    """
    group_size = num_attention_heads // num_key_value_heads
    saved_dtype = q_proj.weight.dtype
    device = q_proj.weight.device

    # Reshape weights
    inp = inp.to(torch.float32)  # [b, l, h]
    WQ = rearrange(q_proj.weight, '(n d) h -> n h d', n=num_attention_heads).to(torch.float32)  # [n, h, d]
    WK = rearrange(k_proj.weight, '(g d) h -> g h d', g=num_key_value_heads).to(torch.float32)  # [g, h, d]

    bias_Q = None
    if q_proj.bias is not None:
        bias_Q = rearrange(q_proj.bias, '(n d) -> n d', n=num_attention_heads).to(torch.float32)  # [n, d]
    
    bias_K = None
    if k_proj.bias is not None:
        bias_K = rearrange(k_proj.bias, '(g d) -> g d', g=num_key_value_heads).to(torch.float32)  # [g, d]

    WQ_trunc_list = []
    WK_trunc_list = []
    bias_Q_list = []
    bias_K_list = []
    indices_list = []

    for g in range(num_key_value_heads):
        s_group = torch.zeros(WQ.shape[2], device=device)  # [d]
        K = inp @ WK[g]  # [B, T, d]
        if bias_K is not None:
            K = K + bias_K[g]  # [B, T, d]
        CK = (K.transpose(1, 2) @ K).mean(dim=0)
        eigvals, eigvecs = torch.linalg.eigh(CK.to(torch.float32))  # For symmetric matrices
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
        CK_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
        # Accumulate group score
        for i in range(group_size):
            idx = g * group_size + i
            Q = inp @ WQ[idx]  # [B, T, d]
            if bias_Q is not None:
                Q = Q + bias_Q[idx]  # [B, T, d]
            if position_embeddings is not None:
                cos, sin = position_embeddings
                Q, K = Q.unsqueeze(1), K.unsqueeze(1) # [b, l, d] -> [b, 1, l, d]
                Q, K = rotary_fn(Q, K, cos, sin)
                Q, K = Q.squeeze(1), K.squeeze(1)
            CQ = (Q.transpose(1, 2) @ Q).mean(dim=0)
            eigvals, eigvecs = torch.linalg.eigh(CQ.to(torch.float32))  # For symmetric matrices
            sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
            CQ_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
            s_group += (CQ_sqrt.norm(dim=0) * CK_sqrt.norm(dim=0)) ** 2

        # rotate half: q = [q_rot, q_pass], k = [k_rot, k_pass]
        if rotate_half:
            # we do the half that is rotated
            s_group = torch.sqrt(s_group)
            quater = WQ.shape[2] // 4 # d // 4
            s_rot_pairs = s_group[:quater] + s_group[quater:2*quater]  # shape [d//4]
            topk_rot_pair_indices = torch.topk(s_rot_pairs, k=rank//4, largest=True).indices  # shape [rank // 4]
            topk_rot = torch.cat([topk_rot_pair_indices, topk_rot_pair_indices + quater], dim=0)  # shape [rank // 2]
            indices_list.append(topk_rot)
            # then we do the half that is not rotated
            half = WQ.shape[2] // 2 # d // 2
            topk_pass = half + torch.topk(s_group[:half], k=rank//2, largest=True).indices  # shape [rank // 2]
            # final concat the two halves
            topk = torch.cat([topk_rot, topk_pass], dim=0)  # shape [rank]
            S_k = torch.eye(WQ.shape[2], device=inp.device, dtype=inp.dtype)[:, topk]  # [d, r]
        else:
            s_group = torch.sqrt(s_group)
            half = WQ.shape[2] // 2 # d // 2
            k_pairs = rank // 2 # rank // 2
            # Compute pair scores
            s_pairs = s_group[:half] + s_group[half:]  # shape [d//2]
            topk_pair_indices = torch.topk(s_pairs, k=k_pairs, largest=True).indices  # shape [k_pairs]
            topk = torch.cat([topk_pair_indices, topk_pair_indices + half], dim=0)  # shape [rank]
            indices_list.append(topk)
            S_k = torch.eye(WQ.shape[2], device=inp.device, dtype=inp.dtype)[:, topk]  # [d, r]


        for i in range(group_size):
            idx = g * group_size + i
            WQ_trunc = WQ[idx] @ S_k  # [h, r]
            WQ_trunc_list.append(WQ_trunc.to(saved_dtype))
            if bias_Q is not None:
                bias_Q_list.append(bias_Q[idx][topk].to(saved_dtype))

        # WK shared within group
        WK_trunc = WK[g] @ S_k  # [h, r]
        WK_trunc_list.append(WK_trunc.to(saved_dtype))
        if bias_K is not None:
            bias_K_list.append(bias_K[g][topk].to(saved_dtype))

    # Concatenate heads
    WQ_trunc = torch.stack(WQ_trunc_list, dim=0) # [n, h, r]
    WK_trunc = torch.stack(WK_trunc_list, dim=0) # [g, h, r]
    WQ_trunc = rearrange(WQ_trunc, 'n h r -> (n r) h')
    WK_trunc = rearrange(WK_trunc, 'g h r -> (g r) h')
    if bias_Q is not None:
        bias_Q_trunc = torch.stack(bias_Q_list, dim=0).flatten() # [n, r]
    if bias_K is not None:
        bias_K_trunc = torch.stack(bias_K_list, dim=0).flatten() # [g, r]

    q_proj_trunc = torch.nn.Linear(WQ_trunc.shape[1], WQ_trunc.shape[0], bias=q_proj.bias is not None, dtype=saved_dtype, device=device)
    q_proj_trunc.weight.data = WQ_trunc
    if bias_Q is not None:
        q_proj_trunc.bias.data = bias_Q_trunc
    k_proj_trunc = torch.nn.Linear(WK_trunc.shape[1], WK_trunc.shape[0], bias=k_proj.bias is not None, dtype=saved_dtype, device=device)
    k_proj_trunc.weight.data = WK_trunc
    if bias_K is not None:
        k_proj_trunc.bias.data = bias_K_trunc

    indices = torch.stack(indices_list, dim=0) # [g, r]
    return q_proj_trunc, k_proj_trunc, indices



from modeling.bamba.bamba_blocks import causal_conv1d_fn
from modeling.bamba.bamba_blocks import BambaRMSNormGated


@torch.no_grad()
def ssd_BC_compression(B_proj, C_proj, dt_proj, inp, rank, intermediate_size, nheads, ngroups, dstate, conv1d):
    """
    Args:
        q_proj: nn.Linear for query projection
        k_proj: nn.Linear for key projection
        dt: [B, T, nheads] time step tensor
        inp: [B, T, d] input tensor to the attention layer
        rank: number of columns to keep (defaults to 70% of per-head dim)
        ngroups: number of groups
        dstate: state size
        conv1d: nn.Conv1d for convolution

    Returns:
        (B_proj_trunc, C_proj_trunc): new compressed weight matrices [num_heads * k, d]
    """
    saved_dtype = B_proj.weight.dtype
    device = B_proj.weight.device
    # Reshape weights per head
    # B_proj.weight [ngroups * dstate, hidden_size]
    # C_proj.weight [ngroups * dstate, hidden_size]
    WB = rearrange(B_proj.weight, '(g d) h -> g h d', g=ngroups).to(torch.float32)  # [g, h, d]
    WC = rearrange(C_proj.weight, '(g d) h -> g h d', g=ngroups).to(torch.float32)  # [g, h, d]

    conv_w_x = conv1d.weight.squeeze(1)[:intermediate_size, :]
    conv_bias_x = conv1d.bias[:intermediate_size]
    conv_weight = conv1d.weight.squeeze(1)[intermediate_size:intermediate_size+2*ngroups*dstate, :]
    conv_bias = conv1d.bias[intermediate_size:intermediate_size+2*ngroups*dstate]

    # Project inputs
    X = inp.to(torch.float32)  # [b, l, h]

    # fp32 dt
    dt_proj = dt_proj.to(torch.float32)
    dt = dt_proj(X)

    WB_trunc_list = []
    WC_trunc_list = []
    Conv_WB_list = []
    Conv_WC_list = []

    for j in range(ngroups):
        WB_j = WB[j]  # [h, d]
        WC_j = WC[j]  # [h, d]

        B_j = X @ WB_j # [b, l, h] @ [h, d] -> [b, l, d]
        C_j = X @ WC_j # [b, l, h] @ [h, d] -> [b, l, d]

        # Apply convolution
        B_C_j = torch.cat([B_j, C_j], dim=2)
        conv_w_bj = conv_weight[j*dstate:(j+1)*dstate, :]
        conv_w_cj = conv_weight[dstate*ngroups+j*dstate:dstate*ngroups+(j+1)*dstate, :]
        conv_w_j = torch.cat([conv_w_bj, conv_w_cj], dim=0)

        conv_bias_bj = conv_bias[j*dstate:(j+1)*dstate]
        conv_bias_cj = conv_bias[dstate*ngroups+j*dstate:dstate*ngroups+(j+1)*dstate]
        conv_bias_j = torch.cat([conv_bias_bj, conv_bias_cj], dim=0)
        B_C_j = causal_conv1d_fn(
            x=B_C_j.transpose(1, 2),
            weight=conv_w_j,
            bias=conv_bias_j,
            activation="silu",
        ).transpose(1, 2)
        B_j = B_C_j[:, :, :dstate]
        C_j = B_C_j[:, :, dstate:]

        CC_j = torch.matmul(C_j.transpose(1, 2), C_j) # [b, d, l] @ [b, l, d] -> [b, d, d]
        CC_j = CC_j.sum(dim=0) / CC_j.shape[0]  # Average over batch, [d, d]
        eigvals, eigvecs = torch.linalg.eigh(CC_j.to(torch.float32))  # For symmetric matrices
        sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
        CC_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

        heads_per_group = nheads // ngroups
        dt_j = dt[:, :, j*heads_per_group:(j+1)*heads_per_group]
        s_group = torch.zeros(B_j.shape[-1], device=device)  # [d]
        # Compute correlation matrices
        for h in range(heads_per_group):
            dB_j = dt_j[:, :, h].unsqueeze(-1) * B_j # [b l 1] * [b l d] -> dB [b l d]
            CB_j = torch.matmul(dB_j.transpose(1, 2), dB_j) # [b, d, l] @ [b, l, d] -> [b, d, d]
            CB_j = CB_j.sum(dim=0) / CB_j.shape[0]  # Average over batch, [d, d]
            eigvals, eigvecs = torch.linalg.eigh(CB_j.to(torch.float32))  # For symmetric matrices
            sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=1e-6))  # Stabilize tiny/negative values
            CB_sqrt = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T
        
            # Compute norm scores
            s_group += (CB_sqrt.norm(dim=0) * CC_sqrt.norm(dim=0))  # [d]

        # Select top-k indices
        k = rank  # or int(0.7 * d)
        topk = torch.topk(s_group, k=k, largest=True).indices
        S_k = torch.eye(dstate, device=X.device, dtype=X.dtype)[:, topk]  # [d, r]

        # Apply compression
        Conv_WB_list.append((conv_w_bj[topk, :], conv_bias_bj[topk]))
        Conv_WC_list.append((conv_w_cj[topk, :], conv_bias_cj[topk]))
        WB_j_trunc = WB_j @ S_k # [h, d] @ [d, r]
        WC_j_trunc = WC_j @ S_k # [h, d] @ [d, r]

        WB_trunc_list.append(WB_j_trunc.to(saved_dtype)) # [h, r]
        WC_trunc_list.append(WC_j_trunc.to(saved_dtype)) # [h, r]

        del CC_sqrt, CB_sqrt, S_k, B_j, C_j, CB_j, CC_j
        torch.cuda.empty_cache()  # Free memory after each head

    # Concatenate all heads, and rearrange
    conv_w_b = torch.cat([conv_w_b for (conv_w_b, _) in Conv_WB_list], dim=0)
    conv_w_c = torch.cat([conv_w_c for (conv_w_c, _) in Conv_WC_list], dim=0)
    conv_bias_b = torch.cat([conv_bias_b for (_, conv_bias_b) in Conv_WB_list], dim=0)
    conv_bias_c = torch.cat([conv_bias_c for (_, conv_bias_c) in Conv_WC_list], dim=0)
    conv_weight = torch.cat([conv_w_x, conv_w_b, conv_w_c], dim=0).to(saved_dtype)
    conv_bias = torch.cat([conv_bias_x, conv_bias_b, conv_bias_c], dim=0).to(saved_dtype)

    conv_kernel_size = conv1d.kernel_size[0]
    compressed_conv1d = torch.nn.Conv1d(
        in_channels=conv_weight.shape[0],
        out_channels=conv_weight.shape[0],
        bias=True,
        kernel_size=conv_kernel_size,
        groups=conv_weight.shape[0],
        padding=conv_kernel_size - 1,
        dtype=saved_dtype,
        device=device,
    )
    compressed_conv1d.weight.data = conv_weight.unsqueeze(1)
    compressed_conv1d.bias.data = conv_bias

    WB_trunc = torch.stack(WB_trunc_list, dim=0) # [n, h, r]
    WC_trunc = torch.stack(WC_trunc_list, dim=0) # [n, h, r]
    WB_trunc = rearrange(WB_trunc, 'n h r -> (n r) h')
    B_proj_trunc = torch.nn.Linear(WB_trunc.shape[1], WB_trunc.shape[0], bias=B_proj.bias is not None, dtype=saved_dtype, device=device)
    B_proj_trunc.weight.data = WB_trunc
    WC_trunc = rearrange(WC_trunc, 'n h r -> (n r) h')
    C_proj_trunc = torch.nn.Linear(WC_trunc.shape[1], WC_trunc.shape[0], bias=C_proj.bias is not None, dtype=saved_dtype, device=device)
    C_proj_trunc.weight.data = WC_trunc
    dt_proj = dt_proj.to(saved_dtype) # cast back to saved_dtype
    return B_proj_trunc, C_proj_trunc, compressed_conv1d



@torch.no_grad()
def ssd_xo_compression(x_proj, out_proj, z_proj, o_inp, rank, intermediate_size, nheads, ngroups,
                       gated_norm, compressed_gated_norm, conv1d, ridge_lambda=1e-4):
    """
    Args:
        x_proj: nn.Linear, the x projection layer
        out_proj: nn.Linear, the out projection layer
        z_proj: nn.Linear, the z projection layer
        o_inp: [B, T, d_h] input features to the output projection, gated normed
        rank: int, target compressed rank
        intermediate_size: int, intermediate size
        nheads: int, number of attention heads
        ngroups: int, number of groups
        gated_norm: the gated normalization layer before the output projection
        compressed_gated_norm: the compressed gated normalization layer
        conv1d: nn.Conv1d for convolution
        ridge_lambda: float, regularization strength λ for ridge leverage

    Returns:
        (x_proj_trunc, out_proj_trunc, z_proj_trunc, compressed_gated_norm, compressed_conv1d): compressed weights
    """
    device = x_proj.weight.device
    saved_dtype = x_proj.weight.dtype

    n = o_inp.shape[0] # [b, l, d_int]
    X = o_inp # [b, l, h] we use fp16 here, and convert to fp32 below batched by batched to save memory
    C_sigma = 0
    for i in range(n):
        Xi = X[i].to(torch.float32) # for numerical stability
        XiT = Xi.t()
        C_sigma += torch.matmul(XiT, Xi) / n # we must use float32 for the numerical stability
    del X, Xi, XiT
    torch.cuda.empty_cache()

    # Ridge leverage score: diag(C (C + λI)^(-1))
    I = torch.eye(C_sigma.shape[0], device=C_sigma.device, dtype=C_sigma.dtype)
    C_ridge_inv = torch.linalg.inv(C_sigma + ridge_lambda * I)
    scores = torch.diag(C_sigma @ C_ridge_inv)  # [d_int]
    # scores = torch.diag(C_sigma)  # [d_int]
    scores_reshaped = scores.reshape(ngroups, nheads // ngroups, -1) # [ngroups, nheads // ngroups, head_dim]
    head_dim = scores_reshaped.shape[-1]
    topk_head_dim = torch.topk(scores_reshaped, k=rank, largest=True, dim=-1).indices # Top-k rank selection
    base_indices = torch.arange(ngroups, device=device) * (nheads // ngroups) * head_dim # [ngroups]
    base_indices = base_indices.unsqueeze(-1) + torch.arange(nheads // ngroups, device=device).unsqueeze(0) * head_dim # [ngroups, nheads // ngroups]
    topk = base_indices.unsqueeze(-1) + topk_head_dim # [ngroups, nheads // ngroups, k]
    topk = topk.reshape(-1) # [ngroups * nheads // ngroups * k]
    Sk = torch.eye(C_sigma.shape[0], device=device)[:, topk]  # [d_int, r]
    # Reconstruct compressed WU and WD
    Wx = x_proj.weight.data.to(torch.float32).t() # [d_int, h] -> [h, d_int]
    Wx_trunc = Wx @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
    Wx_trunc = Wx_trunc.contiguous().to(saved_dtype).t() # [h, r] -> [r, h]
    if z_proj is not None:
        Wz = z_proj.weight.data.to(torch.float32).t() # [d_int, h] -> [h, d_int]
        Wz_trunc = Wz @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
        Wz_trunc = Wz_trunc.contiguous().to(saved_dtype).t()
    else:
        Wz_trunc = None
    Wo = out_proj.weight.data.to(torch.float32).t() # [h, d_int] -> [d_int, h]
    # inv([r, d_int] @ [d_int, d_int] @ [d_int, r]) @ [r, d_int] @ [d_int, d_int] @ [d_int, h]
    Wo_trunc = Sk.T @ Wo
    Wo_trunc = Wo_trunc.contiguous().to(saved_dtype).t()

    x_proj_trunc = torch.nn.Linear(Wx_trunc.shape[1], Wx_trunc.shape[0], bias=x_proj.bias is not None, dtype=saved_dtype, device=device)
    x_proj_trunc.weight.data = Wx_trunc
    if z_proj is not None:
        z_proj_trunc = torch.nn.Linear(Wz_trunc.shape[1], Wz_trunc.shape[0], bias=z_proj.bias is not None, dtype=saved_dtype, device=device)
        z_proj_trunc.weight.data = Wz_trunc
    else:
        z_proj_trunc = None 
    out_proj_trunc = torch.nn.Linear(Wo_trunc.shape[1], Wo_trunc.shape[0], bias=out_proj.bias is not None, dtype=saved_dtype, device=device)
    out_proj_trunc.weight.data = Wo_trunc

    compressed_gated_norm.weight.data = gated_norm.weight.data[topk]

    conv_w_x = conv1d.weight.squeeze(1)[:intermediate_size, :]
    conv_bias_x = conv1d.bias[:intermediate_size]
    conv_w_bc = conv1d.weight.squeeze(1)[intermediate_size:, :]
    conv_bias_bc = conv1d.bias[intermediate_size:]

    conv_weight = torch.cat([conv_w_x[topk, :], conv_w_bc], dim=0)
    conv_bias = torch.cat([conv_bias_x[topk], conv_bias_bc], dim=0)

    conv_kernel_size = conv1d.kernel_size[0]
    compressed_conv1d = torch.nn.Conv1d(
        in_channels=conv_weight.shape[0],
        out_channels=conv_weight.shape[0],
        bias=True,
        kernel_size=conv_kernel_size,
        groups=conv_weight.shape[0],
        padding=conv_kernel_size - 1,
        dtype=saved_dtype,
        device=device,
    )
    compressed_conv1d.weight.data = conv_weight.unsqueeze(1)
    compressed_conv1d.bias.data = conv_bias

    return x_proj_trunc, out_proj_trunc, z_proj_trunc, compressed_gated_norm, compressed_conv1d



from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd



@torch.no_grad()
def ssd_state_aware_xo_compression(x_proj, out_proj, z_proj, inp, B_proj, dt_proj,
                                   reduced_head_dim, intermediate_size, ngroups, nheads, head_dim,
                                   gated_norm, compressed_gated_norm, conv1d, compressed_conv1d,
                                   A_log, chunk_size, dt_bias, dt_softplus, dt_limit, seq_idx, ridge_lambda=1e-4):
    """

    Args:
        x_proj: nn.Linear, the x projection layer
        out_proj: nn.Linear, the out projection layer
        z_proj: nn.Linear, the z projection layer
        inp: [B, T, d_h] input features to the output projection, gated normed
        B_proj: nn.Linear, the B projection layer
        dt_proj: nn.Linear, the dt projection layer
        reduced_head_dim: int, reduced head dimension
        intermediate_size: int, intermediate size
        ngroups: int, number of groups
        nheads: int, number of attention heads
        head_dim: int, head dimension
        gated_norm: the gated normalization layer before the output projection
        compressed_gated_norm: the compressed gated normalization layer
        conv1d: nn.Conv1d for convolution
        compressed_conv1d: nn.Conv1d for compressed convolution
        A_log: [num_heads] or [intermediate_size, state_size] log of A
        chunk_size: int, chunk size
        dt_bias: [num_heads] or [intermediate_size] bias for dt
        dt_softplus: bool, whether to use softplus for dt
        dt_limit: tuple, limit for dt
        seq_idx: [B] or [B, T] sequence index
        ridge_lambda: float, regularization strength λ for ridge leverage

    Returns:
        (x_proj_trunc, out_proj_trunc, z_proj_trunc, compressed_gated_norm, compressed_conv1d): compressed weights
    """
    device = x_proj.weight.device
    saved_dtype = x_proj.weight.dtype
    inp = inp.to(saved_dtype) # to save memory
    # We need to use inp @ weight.data to avoid OOM, x = x_proj(inp) will accumulate 
    # something and cause OOM even with no_grad()...not sure why...
    x = inp @ x_proj.weight.data.t()
    B = inp @ B_proj.weight.data.t()
    dt = inp @ dt_proj.weight.data.t()
    del inp
    torch.cuda.empty_cache()

    _, _, intermediate_size = x.shape
    _, _, ngroups_ssm_state_size = B.shape

    # Apply convolution by using uncompressed conv1d
    xB = torch.cat([x, B], dim=2)
    conv_weight = conv1d.weight.data.squeeze(1)
    conv_bias = conv1d.bias.data
    conv_w_xB = conv_weight[:intermediate_size + ngroups_ssm_state_size, :]
    conv_bias_xB = conv_bias[:intermediate_size + ngroups_ssm_state_size]
    xB = causal_conv1d_fn(
        x=xB.transpose(1, 2),
        weight=conv_w_xB,
        bias=conv_bias_xB,
        activation="silu",
    ).transpose(1, 2)
    x = xB[:, :, :intermediate_size]
    B = xB[:, :, intermediate_size:]
    del xB, conv_w_xB, conv_bias_xB
    torch.cuda.empty_cache()

    # Get scores from state_fwd, use float32 for numerical stability
    batch_size, seq_len, _ = x.shape
    x = x.view(batch_size, seq_len, -1, head_dim).to(torch.float32)
    B = B.view(batch_size, seq_len, ngroups, -1).to(torch.float32)
    dt = dt.to(torch.float32)
    dt_bias = dt_bias.to(torch.float32)
    A = -torch.exp(A_log.float())  # (num_heads) or (intermediate_size, state_size)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states_fwd = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True) # [b, nchunk, nh, hd, dstate]
    del x, B, dt, A, dA_cumsum
    torch.cuda.empty_cache()

    assert not torch.isnan(states_fwd).any(), f"states_fwd has nan"
    states_reshaped = states_fwd.permute(0, 1, 4, 2, 3) # [b, nchunks, nh, hd, dstate] -> [b, nchunks, dstate, hd, nh]
    states_reshaped = states_reshaped.reshape(batch_size, -1, intermediate_size)
    n = states_reshaped.shape[0]
    C_sigma = 0
    for i in range(n):
        Xi = states_reshaped[i].to(torch.float32) # [nchunks*dstate, d_int] for numerical stability
        XiT = Xi.t() # [d_int, nchunks*dstate]
        C_sigma += torch.matmul(XiT, Xi) / n # we must use float32 for the numerical stability
    del states_reshaped, Xi, XiT
    torch.cuda.empty_cache()
    # Ridge leverage score: diag(C (C + λI)^(-1))
    I = torch.eye(C_sigma.shape[0], device=C_sigma.device, dtype=C_sigma.dtype)
    C_ridge_inv = torch.linalg.inv(C_sigma + ridge_lambda * I)
    scores = torch.diag(C_sigma @ C_ridge_inv)  # [d_int]

    scores_reshaped = scores.reshape(ngroups, nheads // ngroups, -1) # [ngroups, nheads // ngroups, head_dim]
    head_dim = scores_reshaped.shape[-1]
    topk_head_dim = torch.topk(scores_reshaped, k=reduced_head_dim, largest=True, dim=-1).indices # Top-k rank selection
    base_indices = torch.arange(ngroups, device=device) * (nheads // ngroups) * head_dim # [ngroups]
    base_indices = base_indices.unsqueeze(-1) + torch.arange(nheads // ngroups, device=device).unsqueeze(0) * head_dim # [ngroups, nheads // ngroups]
    topk = base_indices.unsqueeze(-1) + topk_head_dim # [ngroups, nheads // ngroups, k]
    topk = topk.reshape(-1) # [ngroups * nheads // ngroups * k]
    Sk= torch.eye(nheads*head_dim, device=device, dtype=saved_dtype)[:, topk]

    # Reconstruct compressed WU and WD
    Wx = x_proj.weight.data.t() # [d_int, h] -> [h, d_int]
    Wx_trunc = Wx @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
    Wx_trunc = Wx_trunc.contiguous().to(saved_dtype).t() # [h, r] -> [r, h]
    if z_proj is not None:
        Wz = z_proj.weight.data.t() # [d_int, h] -> [h, d_int]
        Wz_trunc = Wz @ Sk # [h, d_int] @ [d_int, r] -> [h, r]
        Wz_trunc = Wz_trunc.contiguous().to(saved_dtype).t()
    else:
        Wz_trunc = None
    Wo = out_proj.weight.data.t() # [h, d_int] -> [d_int, h]
    Wo_trunc = Sk.T @ Wo
    Wo_trunc = Wo_trunc.contiguous().to(saved_dtype).t()

    x_proj_trunc = torch.nn.Linear(Wx_trunc.shape[1], Wx_trunc.shape[0], bias=x_proj.bias is not None, dtype=saved_dtype, device=device)
    x_proj_trunc.weight.data = Wx_trunc
    if z_proj is not None:
        z_proj_trunc = torch.nn.Linear(Wz_trunc.shape[1], Wz_trunc.shape[0], bias=z_proj.bias is not None, dtype=saved_dtype, device=device)
        z_proj_trunc.weight.data = Wz_trunc
    else:
        z_proj_trunc = None 
    out_proj_trunc = torch.nn.Linear(Wo_trunc.shape[1], Wo_trunc.shape[0], bias=out_proj.bias is not None, dtype=saved_dtype, device=device)
    out_proj_trunc.weight.data = Wo_trunc

    compressed_gated_norm.weight.data = gated_norm.weight.data[topk]
    conv_w_x = compressed_conv1d.weight.squeeze(1)[:intermediate_size, :][topk, :]
    conv_bias_x = compressed_conv1d.bias[:intermediate_size][topk]
    conv_w_bc = compressed_conv1d.weight.squeeze(1)[intermediate_size:, :]
    conv_bias_bc = compressed_conv1d.bias[intermediate_size:]
    conv_weight = torch.cat([conv_w_x, conv_w_bc], dim=0)
    conv_bias = torch.cat([conv_bias_x, conv_bias_bc], dim=0)

    # Create a new compressed conv1d
    conv_kernel_size = compressed_conv1d.kernel_size[0]
    compressed_conv1d = torch.nn.Conv1d(
        in_channels=conv_weight.shape[0],
        out_channels=conv_weight.shape[0],
        bias=True,
        kernel_size=conv_kernel_size,
        groups=conv_weight.shape[0],
        padding=conv_kernel_size - 1,
        dtype=saved_dtype,
        device=device,
    )
    compressed_conv1d.weight.data = conv_weight.unsqueeze(1)
    compressed_conv1d.bias.data = conv_bias

    return x_proj_trunc, out_proj_trunc, z_proj_trunc, compressed_gated_norm, compressed_conv1d