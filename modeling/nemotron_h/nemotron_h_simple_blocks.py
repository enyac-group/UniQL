import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from transformers.utils import logging
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_mamba_2_ssm_available,    
)

logger = logging.get_logger(__name__)

if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined, selective_state_update = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (
        selective_state_update,
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
        causal_conv1d_fn,
        causal_conv1d_update,
    )
)

from .nemotron_h_blocks import HybridMambaAttentionDynamicCache
from .nemotron_h_blocks import reshape_into_chunks, pad_tensor_by_size
from .nemotron_h_blocks import apply_mask_to_padding_states, segment_sum, repeat_kv
from modeling.module_helpers import split_linear, merge_linear
from quantize.quant_linear_layers import HadLinear


class NemotronHMamba2MixerSimple(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, mamba_ssm, use_had_transform=False):
        super().__init__()
        self.config = mamba_ssm.config
        self.num_heads = mamba_ssm.num_heads
        self.hidden_size = mamba_ssm.hidden_size
        self.ssm_state_size = mamba_ssm.ssm_state_size
        self.conv_kernel_size = mamba_ssm.conv_kernel_size
        self.intermediate_size = mamba_ssm.intermediate_size
        self.layer_idx = mamba_ssm.layer_idx
        self.use_conv_bias = mamba_ssm.use_conv_bias
        self.activation = mamba_ssm.activation
        self.act = mamba_ssm.act

        self.layer_norm_epsilon = mamba_ssm.layer_norm_epsilon

        self.n_groups = mamba_ssm.n_groups
        self.head_dim = mamba_ssm.head_dim
        self.chunk_size = mamba_ssm.chunk_size

        self.time_step_limit = mamba_ssm.time_step_limit
        self.time_step_min = mamba_ssm.time_step_min
        self.time_step_max = mamba_ssm.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = mamba_ssm.conv1d

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads

        # # projection of the input hidden states
        # if use_had_transform:
        #     self.in_proj = HadLinear(mamba_ssm.in_proj, input_transform=True, output_transform=False)
        # else:
        #     self.in_proj = mamba_ssm.in_proj

        # Split input projection into multiple linear layers.
        # usually, d_mlp is not used and equal to 0, but we still keep it for compatibility
        d_mlp = (
            mamba_ssm.in_proj.out_features
            - 2 * mamba_ssm.intermediate_size
            - 2 * mamba_ssm.n_groups * mamba_ssm.ssm_state_size
            - mamba_ssm.num_heads
        ) // 2
        split_dims = [
            d_mlp, d_mlp,
            mamba_ssm.intermediate_size, # z
            mamba_ssm.intermediate_size, # x
            mamba_ssm.n_groups * mamba_ssm.ssm_state_size, # B
            mamba_ssm.n_groups * mamba_ssm.ssm_state_size, # C
            mamba_ssm.num_heads, # dt
        ]
        split_names = ["mlp_proj_1", "mlp_proj_2", "z_proj", "x_proj", "B_proj", "C_proj", "dt_proj"]
        linear_list = split_linear(mamba_ssm.in_proj, split_dims)
        for i, (name, dim) in enumerate(zip(split_names, split_dims)):
            if dim <= 0:
                continue
            proj = linear_list.pop(0)
            if use_had_transform:
                proj = HadLinear(proj, input_transform=True, output_transform=False)
            setattr(self, name, proj)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = mamba_ssm.dt_bias

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        self.A_log = mamba_ssm.A_log
        self.norm = mamba_ssm.norm
        self.D = mamba_ssm.D

        # output proj
        if use_had_transform:
            self.out_proj = HadLinear(mamba_ssm.out_proj, input_transform=False, output_transform=True)
        else:
            self.out_proj = mamba_ssm.out_proj
        self.use_bias = mamba_ssm.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def fuse_in_proj(self):
        if not hasattr(self, "in_proj"):
            self.in_proj = merge_linear([self.z_proj, self.x_proj, self.B_proj, self.C_proj, self.dt_proj])
            del self.z_proj, self.x_proj, self.B_proj, self.C_proj, self.dt_proj

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # 1. Gated MLP's linear projection
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        if hasattr(self, "in_proj"):    
            projected_states = self.in_proj(hidden_states)
            gate, hidden_states_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )
        else:
            gate = self.z_proj(hidden_states)
            x = self.x_proj(hidden_states)
            B = self.B_proj(hidden_states)
            C = self.C_proj(hidden_states)
            hidden_states_B_C = torch.cat([x, B, C], dim=-1)
            dt = self.dt_proj(hidden_states)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        # d_mlp = (
        #     projected_states.shape[-1]
        #     - 2 * self.intermediate_size
        #     - 2 * self.n_groups * self.ssm_state_size
        #     - self.num_heads
        # ) // 2

        # Single step calculations via cache
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
            #     [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            # )

            # 2. Convolution sequence transformation
            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )

            # 3. SSM transformation
            A = -torch.exp(self.A_log.float())  # (nheads,)
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)

            # 4. Final linear projection
            out = self.out_proj(hidden_states)[:, None, ...]

        # Fused calculations or step by step if no initialized cache is found
        else:
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

            # 2-4. Fused kernel for conv1d, SSM, and the final projection
            if self.training and cache_params is None:
                projected_states = torch.cat([gate, hidden_states_B_C, dt], dim=-1)
                out = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=False,
                    **dt_limit_kwargs,
                )

            else:
                # _, _, gate, hidden_states_B_C, dt = projected_states.split(
                #     [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
                # )

                # 2. Convolution sequence transformation
                # Init cache
                if cache_params is not None:
                    hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                    conv_states = nn.functional.pad(
                        hidden_states_B_C_transposed,
                        (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                    )
                    cache_params.update_conv_state(
                        layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True
                    )

                if self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                    )
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)
                hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )

                # 3. SSM transformation
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    dt,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )

                # Init cache
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

                scan_output = scan_output.view(batch_size, seq_len, -1)

                # Multiply "gate" branch and apply extra normalization layer
                scan_output_gatednorm = self.norm(scan_output, gate)

                # 4. Final linear projection
                out = self.out_proj(scan_output_gatednorm)
        # return out, scan_output, gate, scan_output_gatednorm
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[HybridMambaAttentionDynamicCache]=None, cache_position:Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        if hasattr(self, "in_proj"):
            projected_states = self.in_proj(input_states)
            gate, hidden_states_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )
        else:
            gate = self.z_proj(hidden_states)
            # hidden_states_B_C = self.xBC_proj(hidden_states)
            x = self.x_proj(hidden_states)
            B = self.B_proj(hidden_states)
            C = self.C_proj(hidden_states)
            hidden_states_B_C = torch.cat([x, B, C], dim=-1)
            dt = self.dt_proj(hidden_states)

        # d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size - 2 * self.n_groups * self.ssm_state_size-self.num_heads) // 2
        # _, _, gate, hidden_states_B_C, dt = projected_states.split(
        #         [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        # )

        # 2. Convolution sequence transformation
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)

            # We need to guarantee that anything regarding the cache is on the same device
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)

            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = cache_params.ssm_states.device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            # State calculation
            cache_params.update_ssm_state(
                layer_idx=self.layer_idx,
                new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # Contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            # (right term of low-rank factorization of off-diagonal blocks; B terms)
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # (middle term of factorization of off-diag blocks; A terms)
            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # 4. Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[HybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
        if is_fast_path_available and "cuda" in next(self.parameters()).device.type: # more general and works for SVDLinear
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)



class NemotronHAttentionSimple(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attn, use_had_transform = True):
        super().__init__()
        self.config = attn.config
        self.layer_idx = attn.layer_idx
        if self.layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = attn.attention_dropout
        self.hidden_size = attn.hidden_size
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.num_key_value_heads = attn.num_key_value_heads
        self.num_key_value_groups = attn.num_key_value_groups
        self.max_position_embeddings = attn.max_position_embeddings
        self.is_causal = True

        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        # output proj
        if use_had_transform:
            self.q_proj = HadLinear(attn.q_proj, input_transform=True, output_transform=False)
            self.k_proj = HadLinear(attn.k_proj, input_transform=True, output_transform=False)
            self.v_proj = HadLinear(attn.v_proj, input_transform=True, output_transform=False)
            self.o_proj = HadLinear(attn.o_proj, input_transform=False, output_transform=True)
        else:
            self.q_proj = attn.q_proj
            self.k_proj = attn.k_proj
            self.v_proj = attn.v_proj
            self.o_proj = attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_embeddings: Tuple[torch.Tensor, torch.Tensor], #TODO
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        #attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



class NemotronHMLPSimple(nn.Module):
    def __init__(self, mlp, use_had_transform = True):
        super().__init__()
        self.config = copy.deepcopy(mlp.config)
        self.layer_idx = mlp.layer_idx
        if self.layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size

        if use_had_transform:
            self.up_proj = HadLinear(mlp.up_proj, input_transform=True, output_transform=False)
            self.down_proj = HadLinear(mlp.down_proj, input_transform=False, output_transform=True)
        else:
            self.up_proj = mlp.up_proj
            self.down_proj = mlp.down_proj
        
        self.act_fn = mlp.act_fn

    def forward(self, x):
        x = self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x
    