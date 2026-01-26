# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
from megatron.core.utils import deprecate_inference_params, log_single_rank, make_tp_sharded_tensor_for_checkpoint

try:
    from dragon_mamba3_ops.mimo_variant.ssd_mimo import mamba_chunk_scan_discretized_fused_combined as mamba_mimo_chunk_scan_discretized_fused_combined
    from dragon_mamba3_ops.angle_cumsum import angle_dt
    from dragon_mamba3_ops.rotary_mamba_mimo import rotary_qk as mimo_rotary_qk
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


logger = logging.getLogger(__name__)


class ExtendedEmbedding(torch.nn.Embedding):
    """
    torch.nn.Embedding with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1 (embedding dim)."""        
        state_dict = self.state_dict(prefix="", keep_vars=True)
        weight_prefix = f"{prefix}weight"
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict["weight"],
                key=weight_prefix,
                tp_axis=1,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
            )
        }

class ExtendedRMSNorm(RMSNormGated):
    """
    RMSNormGated with sharded state dict.
    """

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0}, sharded_offsets
        )

@dataclass
class Mamba3Submodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    b_norm: Union[ModuleSpec, type] = None
    c_norm: Union[ModuleSpec, type] = None
    rope_proj: Union[ModuleSpec, type] = None

class Mamba3(MegatronModule):
    def __init__(
        self,
        config: DragonConfig,
        submodules: Mamba3Submodules,
        layer_number: int,
        vocab_size: int = 50000,
        use_ve: bool = False,
        input_scalar: float = 1.,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        if not HAVE_EINOPS:
            raise ImportError("einops is required by the Mamba model but cannot be imported")

        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        self.d_model = config.hidden_size
        self.d_inner = int(2 * self.d_model)
        self.rope_fraction = self.config.mamba_rope_fraction
        self.A_floor = 1e-4
        self.chunk_size = 128
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
        assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 2 * self.ngroups * self.d_state * self.mimo_dim + 3 * self.nheads,  # z x B C dt A trap
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            return_layernorm_output=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="in_proj",
            tp_group=self.pg_collection.tp,
            alpha_fwd=input_scalar,
            alpha_bwd=input_scalar,
        )
        # WARNING: A_proj was specified as "float32". here, we merge it with in_proj so it's no longer float32.

        # VE embeddings and scalars
        self.use_ve = use_ve
        if use_ve:
            self.ve_embedding = ExtendedEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=self.ngroups_local_tp*self.d_inner_per_group,
            )
            with torch.no_grad():
                self.ve_embedding.weight.normal_(mean=0.0, std=config.init_embedding_std)
            setattr(self.ve_embedding.weight, 'tensor_model_parallel', True)
            self.ve_scalars = torch.nn.Parameter(torch.zeros(self.ngroups_local_tp, self.d_inner_per_group, dtype=torch.float32))
            setattr(self.ve_scalars, 'tensor_model_parallel', True)

        self.rope_proj = build_module(
            submodules.rope_proj,
            self.config.hidden_size,
            self.num_rope_angles,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode='duplicated',
            is_expert=False,
            tp_comm_buffer_name='rope_proj',
            alpha_fwd=input_scalar,
            alpha_bwd=input_scalar,
        )
        w = self.rope_proj.weight
        b = getattr(self.rope_proj, "bias", None)
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

        self.B_bias = nn.Parameter(torch.ones((self.mimo_dim, self.nheads_local_tp, self.d_state)), requires_grad=True)
        self.C_bias = nn.Parameter(torch.ones((self.mimo_dim, self.nheads_local_tp, self.d_state)), requires_grad=True)
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
        in_proj_mimo_x_init_weights = torch.ones(self.dr_out_dim_local_tp, self.mimo_dim*self.mimo_proj_block_order, self.mimo_proj_block_order)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.dr_out_dim_local_tp, self.mimo_dim*self.mimo_proj_block_order, self.mimo_proj_block_order)
        out_proj_mimo_init_weights = torch.ones(self.dr_out_dim_local_tp, self.mimo_proj_block_order, self.mimo_dim*self.mimo_proj_block_order)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)
        setattr(self.in_proj_mimo_x, "tensor_model_parallel", True)
        setattr(self.in_proj_mimo_z, "tensor_model_parallel", True)
        setattr(self.out_proj_mimo, "tensor_model_parallel", True)

        with get_cuda_rng_tracker().fork():
        #with nullcontext():
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

        self.output_norm = ExtendedRMSNorm(
            self.d_inner_local_tp,
            eps=config.layernorm_epsilon,
            group_size=self.d_inner_local_tp // self.ngroups_local_tp,
            norm_before_gate=False,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.output_norm.weight, "tp_sync", True)

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
        input_ids: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        in_inference_mode = inference_context is not None and not self.training
        assert not in_inference_mode

        # Input projection
        out, _ = self.in_proj(hidden_states)
        zxBCdtAtrap, normed_hidden_states = out
        zxBCdtAtrap = zxBCdtAtrap.transpose(0, 1) # s b x --> b s x
        zxBCdtAtrap = rearrange(zxBCdtAtrap, "b l (G D) -> b l G D", G=self.ngroups_local_tp)#.contiguous()
        # split per group: [B, L, G_local, D_group]
        z = zxBCdtAtrap[..., 0:self.d_inner_per_group]; accum = self.d_inner_per_group
        x = zxBCdtAtrap[..., accum:accum+self.d_inner_per_group]; accum += self.d_inner_per_group
        B = zxBCdtAtrap[..., accum:accum+self.d_state*self.mimo_dim]; accum += self.d_state*self.mimo_dim
        C = zxBCdtAtrap[..., accum:accum+self.d_state*self.mimo_dim]; accum += self.d_state*self.mimo_dim
        dt = zxBCdtAtrap[..., accum:accum+self.nheads_per_group]; accum += self.nheads_per_group
        A = zxBCdtAtrap[..., accum:accum+self.nheads_per_group]; accum += self.nheads_per_group
        trap = zxBCdtAtrap[..., accum:accum+2*self.nheads_per_group]

        z = rearrange(z, "b l G d -> b l (G d)")
        x = rearrange(x, "b l G d -> b l (G d)")
        B = rearrange(B, "b l G d -> b l (G d)")
        C = rearrange(C, "b l G d -> b l (G d)")
        dt = rearrange(dt, "b l G n -> b l (G n)")
        A = rearrange(A, "b l G n -> b l (G n)")
        trap = rearrange(trap, "b l G n -> b l (G n)")

        # value embeddings
        if self.use_ve:
            ve = self.ve_embedding(input_ids) # (B,S, G_local*D)
            x = x + self.ve_scalars.view(1, 1, -1) * ve

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)

        # Perform MIMO x and z up projection (d_inner -> mimo_rank*d_inner)
        x = rearrange(x, "b l (d g) -> b l d g", g=self.mimo_proj_block_order)
        x = torch.einsum("bldg,drg->blrd", x, self.in_proj_mimo_x)

        z = rearrange(z, "b l (d g) -> b l d g", g=self.mimo_proj_block_order)
        z = torch.einsum("bldg,drg->blrd", z, self.in_proj_mimo_z)

        if self.mimo_proj_block_order > 1:
            x = rearrange(x, "b l g d -> b l (g d)")
            x = rearrange(x, "b l (r d) -> b l r d", r=self.mimo_dim)
            z = rearrange(z, "b l g d -> b l (g d)")
            z = rearrange(z, "b l (r d) -> b l r d", r=self.mimo_dim)
        
        x = rearrange(x, "b l r (h p) -> b l r h p", p=self.headdim)

        B = rearrange(B, "b l (g r n) -> b l r g n", g=self.ngroups_local_tp, r=self.mimo_dim)
        C = rearrange(C, "b l (g r n) -> b l r g n", g=self.ngroups_local_tp, r=self.mimo_dim)    

        B = self.B_norm(B)
        C = self.C_norm(C)

        if self.ngroups != self.nheads:
            n_repeat = self.nheads_local_tp // self.ngroups_local_tp
            assert self.nheads_local_tp % self.ngroups_local_tp == 0
            B = B.repeat(1, 1, 1, n_repeat, 1) # (B, L, R, N, S)
            C = C.repeat(1, 1, 1, n_repeat, 1) # (B, L, R, N, S)

        angle, _ = self.rope_proj(normed_hidden_states) # (L, B, S)
        if self.config.sequence_parallel:
            angle = gather_from_sequence_parallel_region(angle, group=self.pg_collection.tp)
        angle = angle.transpose(0, 1) # (B, L, S)
        angle = angle.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1) # (B, L, G, S)
        angle = angle_dt(angle, dt)

        C, B, CB_sum = mimo_rotary_qk(q=C, k=B, angle=angle, bias_q=self.C_bias, bias_k=self.B_bias, conjugate=False, inplace=False)

        A = _A * dt
        gating_factor = dt # B, L, N

        trap = F.sigmoid(trap) # (B, L, N)

        alpha_arr = torch.exp(A)
        beta_arr = (1-trap)*gating_factor*alpha_arr
        gamma_arr = trap*gating_factor

        # roll alpha and beta to the left by 1
        _alpha_arr = torch.roll(alpha_arr, shifts=-1, dims=1)
        _beta_arr = torch.roll(beta_arr, shifts=-1, dims=1)

        x_scalar = (gamma_arr*_alpha_arr + _beta_arr).to(torch.bfloat16)

        y = mamba_mimo_chunk_scan_discretized_fused_combined(
            x=x.bfloat16(),
            A=A.bfloat16(),
            B=B.bfloat16(),
            C=C.bfloat16(),
            chunk_size=self.chunk_size,
            x_scalar=x_scalar,
            gamma=gamma_arr,
            CB_sum=CB_sum,
            D=self.D,
            z=None,
        )

        y = rearrange(y, "b l r h p -> b l r (h p)")
        y = self.output_norm(y, z)

        # Perform MIMO down projection (mimo_rank*d_inner -> d_inner)
        y = rearrange(y, "b l r d -> b l (r d)")
        y = rearrange(y, "b l (g d) -> b l g d", g=self.mimo_dim*self.mimo_proj_block_order)
        y = torch.einsum("blgd,drg->bldr", y, self.out_proj_mimo.to(y.dtype))
        y = rearrange(y, "b l d r -> b l (d r)")
        y = rearrange(y, "b l (h d) -> b l h d", d=self.headdim)

        y = y.transpose(0, 1).contiguous() # b s h d -> s b h d

        return y

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        axis_map = {
            "dt_bias": 0,
            "D": 0,
            "B_bias": 1,
            "C_bias": 1,
            "in_proj_mimo_x": 0,
            "in_proj_mimo_z": 0,
            "out_proj_mimo": 0,
        }
        if self.use_ve:
            axis_map.update({
                've_scalars': 0,
            })
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
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
