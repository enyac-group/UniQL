import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def apply_rotary_pos_emb_indices(q, k, cos, sin, indices, unsqueeze_dim=1):

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    reduced_rotary_dim = indices.shape[-1]
    q_rot, q_pass = q[..., :reduced_rotary_dim], q[..., reduced_rotary_dim:]
    k_rot, k_pass = k[..., :reduced_rotary_dim], k[..., reduced_rotary_dim:]

    bszie, num_attention_heads, seqlen, _ = q_rot.shape
    bszie, num_key_value_heads, seqlen, _ = k_rot.shape

    if num_attention_heads == num_key_value_heads:
        index_reshaped = indices.view(1, indices.shape[0], 1, indices.shape[1])
        index_expanded = index_reshaped.expand((bszie, num_attention_heads, seqlen, reduced_rotary_dim))

        # torch.gather only supports int64 indices
        if index_expanded.dtype != torch.int64:
            index_expanded = index_expanded.to(torch.int64)

        cos_expand = cos.unsqueeze(unsqueeze_dim).expand(bszie, num_attention_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        sin_expand = sin.unsqueeze(unsqueeze_dim).expand(bszie, num_attention_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        q_cos_reduced = torch.gather(cos_expand, 3, index_expanded)
        q_sin_reduced = torch.gather(sin_expand, 3, index_expanded)
        k_cos_reduced = q_cos_reduced
        k_sin_reduced = q_sin_reduced

    else:
        assert indices.shape[0] == num_key_value_heads, "indices shape mismatch"
        index_reshaped = indices.view(1, indices.shape[0], 1, indices.shape[1])
        k_index_expanded = index_reshaped.expand((bszie, num_key_value_heads, seqlen, reduced_rotary_dim))
        q_index_expanded = torch.repeat_interleave(k_index_expanded, num_attention_heads // num_key_value_heads, dim=1)

        # torch.gather only supports int64 indices
        if k_index_expanded.dtype != torch.int64:
            k_index_expanded = k_index_expanded.to(torch.int64)
        if q_index_expanded.dtype != torch.int64:
            q_index_expanded = q_index_expanded.to(torch.int64)
        
        q_cos_expand = cos.unsqueeze(unsqueeze_dim).expand(bszie, num_attention_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        q_sin_expand = sin.unsqueeze(unsqueeze_dim).expand(bszie, num_attention_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        q_cos_reduced = torch.gather(q_cos_expand, 3, q_index_expanded)
        q_sin_reduced = torch.gather(q_sin_expand, 3, q_index_expanded)
        k_cos_expand = cos.unsqueeze(unsqueeze_dim).expand(bszie, num_key_value_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        k_sin_expand = sin.unsqueeze(unsqueeze_dim).expand(bszie, num_key_value_heads, seqlen, rotary_dim) # [1, 2048, 128] -> [1, 1, 2048, 128]
        k_cos_reduced = torch.gather(k_cos_expand, 3, k_index_expanded)
        k_sin_reduced = torch.gather(k_sin_expand, 3, k_index_expanded)

    q_embed = (q_rot * q_cos_reduced) + (rotate_half(q_rot) * q_sin_reduced)
    k_embed = (k_rot * k_cos_reduced) + (rotate_half(k_rot) * k_sin_reduced)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
