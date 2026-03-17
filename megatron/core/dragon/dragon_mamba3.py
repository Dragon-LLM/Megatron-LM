# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, log_single_rank, make_tp_sharded_tensor_for_checkpoint, nvtx_range_pop, nvtx_range_push


try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False

try:
    from dragon_mamba3_fast.fused_mimo_variant.mamba3_tilelang import mamba3_tilelang
    from dragon_mamba3_fast.angle_cumsum import angle_dt
    HAVE_FAST_MAMBA_SSM = True
except ImportError as e:
    HAVE_FAST_MAMBA_SSM = True
    raise e

logger = logging.getLogger(__name__)

@dataclass
class Mamba3Submodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    b_norm: Union[ModuleSpec, type] = None
    c_norm: Union[ModuleSpec, type] = None
    rope_proj: Union[ModuleSpec, type] = None
    output_norm: Union[ModuleSpec, type] = None
    dyn_proj: Union[ModuleSpec, type] = None

class FastMamba3(MegatronModule):
    def __init__(
        self,
        config: DragonConfig,
        submodules: Mamba3Submodules,
        layer_number: int,
        input_scalar: float = 1.,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        self.d_model = config.hidden_size
        self.d_inner = int(2 * self.d_model)
        self.rope_fraction = self.config.mamba_rope_fraction
        if self.rope_fraction == 0.5:
            self.rotary_dim_divisor = 4
        elif self.rope_fraction == 1.0:
            self.rotary_dim_divisor = 2
        else:
            raise ValueError(f"rope fraction of {self.rope_fraction} is currently not supported.")

        self.A_floor = 1e-4
        self.chunk_size = 64 // self.config.mamba_mimo_dim
        assert pg_collection is not None, "pg_collection must be provided for Mamba3"
        self.pg_collection = pg_collection

        self.use_mem_eff_path = self.config.use_mamba_mem_eff_path
        if not self.use_mem_eff_path:
            log_single_rank(
                logger,
                logging.WARNING,
                (
                    "We are not currently using or functionally testing use_mem_eff_path==False "
                    "for training. It may not work as expected."
                ),
            )

        self.d_state = self.config.mamba_state_dim
        self.headdim = self.config.mamba_head_dim
        self.ngroups = self.config.mamba_num_groups
        self.mimo_dim = self.config.mamba_mimo_dim
        self.mimo_proj_block_order = self.config.mamba_mimo_proj_block_order

        assert self.ngroups == 1
        assert self.d_state is not None and self.d_state > 0
        assert self.headdim is not None and self.headdim > 0
        assert self.ngroups is not None and self.ngroups > 0

        if self.config.mamba_num_heads is not None:
            self.nheads = self.config.mamba_num_heads
            assert self.nheads > 0
            self.d_inner = self.nheads * self.headdim
        else:
            assert self.d_inner % self.headdim == 0, "d_inner must be evenly divisible by headdim"
            self.nheads = self.d_inner // self.headdim
        self.dr_out_dim = self.d_inner // self.mimo_proj_block_order

        self.split_tensor_size = int(self.d_state * self.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        if self.config.fp8:
            assert (2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads) % 16 == 0, (
                "For FP8, the innermost dimension of the Mamba layer "
                "input projection output tensor must be a multiple of 16."
            )

        tp_size = self.pg_collection.tp.size()

        # Ensure that each TP rank gets at least one head:
        assert self.nheads % tp_size == 0, "nheads must be evenly divisble by tp_size"
        self.nheads_per_group = self.nheads // self.ngroups
        self.nheads_local_tp = self.nheads // tp_size

        # Note that we do not need to confirm that `d_inner % tp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = d_inner // headdim`, and `nheads % tp_size == 0`
        self.d_inner_per_group = self.d_inner // self.ngroups
        self.d_inner_local_tp = self.d_inner // tp_size
        self.dr_out_dim_local_tp = self.dr_out_dim // tp_size

        # Ensure that each TP rank gets at least one group:
        #assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        if not self.config.use_geodesic_norm:
            args = {
                "alpha_fwd": input_scalar,
                "alpha_bwd": input_scalar,
                "return_layernorm_output": True,
            }
        else:
            args = {
                "alpha_fwd": input_scalar,
                "alpha_bwd": input_scalar,
            }

        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 3 * self.nheads,  # z x dt A trap angle
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="in_proj",
            tp_group=self.pg_collection.tp,
            **args,
        )

        args = {
            "alpha_fwd": input_scalar,
            "alpha_bwd": input_scalar,
        }
        self.in_proj_dyn = build_module(
            submodules.dyn_proj,
            self.config.hidden_size,
            2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles, # B C angle
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode='duplicated',
            is_expert=False,
            tp_comm_buffer_name='in_proj_dyn',
            **args,
        )
        w = self.in_proj_dyn.weight
        b = getattr(self.in_proj_dyn, "bias", None)
        if self.config.sequence_parallel:
            # doesnt see the same data => sum grads
            setattr(w, 'tp_sync', True)
            if b is not None:
                setattr(b, 'tp_sync', True)
        else:
            # see the same data => avg grads
            setattr(w, 'average_gradients_across_tp_domain', True)
            if b is not None:
                setattr(b, 'average_gradients_across_tp_domain', True)

        self.B_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.C_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        setattr(self.B_bias, "tensor_model_parallel", True)
        setattr(self.C_bias, "tensor_model_parallel", True)

        self.B_norm = build_module(
            submodules.b_norm,
            hidden_size=self.d_state,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.C_norm = build_module(
            submodules.c_norm,
            hidden_size=self.d_state,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        setattr(self.B_norm.weight, "tp_sync", True)
        setattr(self.C_norm.weight, "tp_sync", True)

        # Initialize up/down MIMO projection (for x and z)
        in_proj_mimo_x_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)
        out_proj_mimo_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)
        setattr(self.in_proj_mimo_x, "tensor_model_parallel", True)
        setattr(self.in_proj_mimo_z, "tensor_model_parallel", True)
        setattr(self.out_proj_mimo, "tensor_model_parallel", True)

        with get_cuda_rng_tracker().fork():
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local_tp,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
            setattr(self.dt_bias, "tensor_model_parallel", True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads_local_tp, device=torch.cuda.current_device())) # Keep in fp32
        self.D._no_weight_decay = True # useless flag
        setattr(self.D, "tensor_model_parallel", True)

        self.output_norm = build_module(
            submodules.output_norm,
            hidden_size=self.d_inner,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        w = getattr(self.output_norm, "weight", None)
        if w is not None:
            w.tp_sync = True

        self.window_size = 0

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        window_size: Optional[Tuple[int, int]] = None, # not used, for compatibility
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """

        #self._maintain_float32_params()
        inference_context = deprecate_inference_params(inference_context, inference_params)
        in_inference_mode = inference_context is not None and not self.training
        assert not in_inference_mode

        # Input projection
        nvtx_range_push(suffix="M3_input_proj")
        out, _ = self.in_proj(hidden_states)

        if self.config.use_geodesic_norm:
            zxdtAtrap = out
            normed_hidden_states = hidden_states
        else:
            zxdtAtrap, normed_hidden_states = out
        
        BCangle, _ = self.in_proj_dyn(normed_hidden_states)

        if self.config.artificial_seq_len > 0:
            seq_len, batch_size, dim = zxdtAtrap.shape
            artificial_batch_size = int(batch_size * (seq_len // self.config.artificial_seq_len))
            #print("Reshaping zxdtAtrap for complete SLW: ", zxdtAtrap.shape, "->", (self.config.artificial_seq_len, artificial_batch_size, dim))
            zxdtAtrap = zxdtAtrap.reshape(self.config.artificial_seq_len, artificial_batch_size, dim)

        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        off = 0
        z    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        x    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        dt   = per_head[..., off];                        off += 1           # (L, B, H)
        A    = per_head[..., off];                        off += 1           # (L, B, H)
        trap = per_head[..., off];                        off += 1           # (L, B, H)
        z = rearrange(z, "l b H p -> b l H p")
        x = rearrange(x, "l b H p -> b l H p")
        dt   = rearrange(dt, "l b n -> b l n").to(torch.float32)
        A    = rearrange(A, "l b n -> b l n")
        trap = rearrange(trap, "l b n -> b n l")

        off = 0
        B     = BCangle[..., off : off + self.ngroups*self.mimo_dim*self.d_state]; off += self.ngroups*self.mimo_dim*self.d_state
        C     = BCangle[..., off : off + self.ngroups*self.mimo_dim*self.d_state]; off += self.ngroups*self.mimo_dim*self.d_state
        angle = BCangle[..., off :]
        B = rearrange(B, "l b (G r n) -> l b r G n", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "l b (G r n) -> l b r G n", G=self.ngroups, r=self.mimo_dim)
        nvtx_range_pop(suffix="M3_input_proj")

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)
        ADT = _A * dt

        nvtx_range_push(suffix="M3_mimo_BC_norm")
        B = self.B_norm(B)
        C = self.C_norm(C)

        if self.config.sequence_parallel:
            B = gather_from_sequence_parallel_region(B, group=self.pg_collection.tp)
            C = gather_from_sequence_parallel_region(C, group=self.pg_collection.tp)
            angle = gather_from_sequence_parallel_region(angle, group=self.pg_collection.tp)

        if self.config.artificial_seq_len > 0:
            #print("Reshaping B and C back for complete SLW: ", B.shape, "->", (self.config.artificial_seq_len, artificial_batch_size, *B.shape[2:]), " and ", C.shape, "->", (self.config.artificial_seq_len, artificial_batch_size, *C.shape[2:]))
            B = B.reshape(self.config.artificial_seq_len, artificial_batch_size, *B.shape[2:])
            C = C.reshape(self.config.artificial_seq_len, artificial_batch_size, *C.shape[2:])
            angle = angle.reshape(self.config.artificial_seq_len, artificial_batch_size, *angle.shape[2:])

        B = rearrange(B, "l b r G n -> b l r G n").contiguous()
        C = rearrange(C, "l b r G n -> b l r G n").contiguous()
        a, b, c, d, e = C.size()
        C = C.as_strided(size=(a, b, c, d, e), stride=(b*c*d*e, c*d*e, d*e, e, 1))
        a, b, c, d, e = B.size()
        B = B.as_strided(size=(a, b, c, d, e), stride=(b*c*d*e, c*d*e, d*e, e, 1))
        nvtx_range_pop(suffix="M3_mimo_BC_norm")

        angle = angle.transpose(0, 1) # (B, L, S) 
        angle = angle.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1) # (B, L, G, S)
        angle = angle_dt(angle, dt)

        ADT = rearrange(ADT, "b l n -> b n l")
        dt = rearrange(dt, "b l n -> b n l")

        y = mamba3_tilelang(
            Q=C.contiguous(),
            K=B.contiguous(),
            V=x.contiguous(),
            ADT=ADT.to(torch.float32).contiguous(),
            DT=dt.to(torch.float32).contiguous(),
            Trap=trap.contiguous(),
            Q_bias=self.C_bias.to(torch.float32),
            K_bias=self.B_bias.to(torch.float32),
            MIMO_V=self.in_proj_mimo_x.to(torch.float32),
            MIMO_Z=self.in_proj_mimo_z.to(torch.float32),
            MIMO_Out=self.out_proj_mimo.to(torch.float32),
            Angles=angle.to(torch.float32).contiguous(),
            D=self.D.to(torch.float32).contiguous(),
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
        )
        nvtx_range_pop(suffix="M3_mimo_chunk_scan")

        y = rearrange(y, "b l h p -> l b (h p)")

        y = self.output_norm(y)

        if self.config.artificial_seq_len > 0:
            y = y.reshape(seq_len, batch_size, -1)

        return y, normed_hidden_states

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        
        axis_map = {
            "dt_bias": 0,
            "D": 0,
            "B_bias": 0,
            "C_bias": 0,
            "in_proj_mimo_x": 0,
            "in_proj_mimo_z": 0,
            "out_proj_mimo": 0,
        }

        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map=axis_map, # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata
            )

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict
    
    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)
