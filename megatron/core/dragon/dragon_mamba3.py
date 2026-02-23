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
from megatron.core.utils import deprecate_inference_params, log_single_rank, make_tp_sharded_tensor_for_checkpoint, nvtx_range_pop, nvtx_range_push
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory, ReplicaId

try:
    from dragon_mamba3_ops.mimo_variant.ssd_mimo import mamba_chunk_scan_discretized_fused_combined as mamba_mimo_chunk_scan_discretized_fused_combined
    from dragon_mamba3_ops.angle_cumsum import angle_dt
    from dragon_mamba3_ops.rotary_mamba_mimo import rotary_qk as mimo_rotary_qk
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

    HAVE_MAMBA_SSM = True
except ImportError:
    HAVE_MAMBA_SSM = False

try:
    from dragon_mamba3_fast.fused_mimo_variant.mamba3_tilelang import mamba3_tilelang
    if not HAVE_MAMBA_SSM:
       from dragon_mamba3_fast.angle_cumsum import angle_dt
    HAVE_FAST_MAMBA_SSM = True
except ImportError:
    from dragon_mamba3_ops.fused_mimo_variant.mamba3_tilelang import mamba3_tilelang
    if not HAVE_MAMBA_SSM:
       from dragon_mamba3_ops.angle_cumsum import angle_dt

    HAVE_FAST_MAMBA_SSM = True
    print("dragon_mamba3_fast not found")

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
    output_norm: Union[ModuleSpec, type] = None
    dyn_proj: Union[ModuleSpec, type] = None


class Dynamic_erf(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # --- FIX START ---
        # Mark these as Tensor Parallel so the optimizer knows they are split
        setattr(self.weight, "tensor_model_parallel", True)
        setattr(self.bias, "tensor_model_parallel", True)
        
        # Alpha and Shift are global scalars. They should NOT be marked tensor_model_parallel.
        # Instead, we mark them as tp_sync so gradients are summed across TP ranks.
        setattr(self.alpha, "tp_sync", True)
        setattr(self.shift, "tp_sync", True)
        # --- FIX END ---

    #@torch.compile
    def forward(self, x):
        return self.weight * torch.erf(self.alpha * x + self.shift) + self.bias

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}'

class FastMamba3(MegatronModule):
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

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 3 * self.nheads,  # z x B C dt A trap #angle
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
        self.dim_dyn_output = 2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles
        self.in_proj_dyn = build_module(
            submodules.dyn_proj,
            self.config.hidden_size,
            self.dim_dyn_output,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode='duplicated',
            is_expert=False,
            tp_comm_buffer_name='in_proj_dyn',
            alpha_fwd=input_scalar,
            alpha_bwd=input_scalar,
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
    
        self.use_ve = use_ve
        if use_ve:
            self.ve_embedding = ExtendedEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=self.ngroups_local_tp*self.d_inner_per_group,
            )
            with torch.no_grad():
                self.ve_embedding.weight.normal_(mean=0.0, std=config.init_embedding_std)
            setattr(self.ve_embedding.weight, 'tensor_model_parallel', True)
            self.ve_scalars = torch.nn.Parameter(torch.zeros(self.ngroups_local_tp, self.d_inner_per_group)) #, dtype=torch.float32))
            setattr(self.ve_scalars, 'tensor_model_parallel', True)

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
        # print("Mamba Head dim: ", self.headdim)
        in_proj_mimo_x_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)
        out_proj_mimo_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)
        setattr(self.in_proj_mimo_x, "tensor_model_parallel", True)
        setattr(self.in_proj_mimo_z, "tensor_model_parallel", True)
        setattr(self.out_proj_mimo, "tensor_model_parallel", True)

        #with get_cuda_rng_tracker().fork():
        with nullcontext():
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

        #self.output_norm = Dynamic_erf(
        #    normalized_shape=self.d_inner_local_tp,
        #)
        #self.output_norm = torch.compile(self.output_norm)
        
        self.output_norm=build_module(
            submodules.output_norm,
            hidden_size=self.d_inner,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        setattr(self.output_norm.weight, "tp_sync", True)
        
        self.n_repeat = self.nheads_local_tp // self.ngroups
        #print(f"Repeating B and C for n_repeat={n_repeat} because ngroups={self.ngroups} != nheads={self.nheads}")

        # In TPFastMamba3.__init__, at the very end:
        #print("\n=== DEBUG: TPFastMamba3 Parameters ===")
        #for name, p in self.named_parameters():
        #    print(f"Name: {name} | Size: {p.shape}")
        #    if name == 'param':
        #        print("!!! FOUND INVALID PARAMETER 'param' !!!")
        #print("======================================\n")


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

        #self._maintain_float32_params()
        #print(f"Mamba3 forward pass with TPFast: hidden_states shape {hidden_states.shape}")
        inference_context = deprecate_inference_params(inference_context, inference_params)
        in_inference_mode = inference_context is not None and not self.training
        assert not in_inference_mode

        # Input projection
        nvtx_range_push(suffix="M3_input_proj")
        out, out2 = self.in_proj(hidden_states)
        zxdtAtrap, normed_hidden_states = out
        #print(f"After in_proj: zxdtAtrap shape {zxdtAtrap.shape}, normed_hidden_states shape {normed_hidden_states.shape}")
        BCangle, _ = self.in_proj_dyn(normed_hidden_states)
        #print(f"After in_proj_dyn: BC shape {BC.shape}")
        offset = 0
        z = zxdtAtrap[..., offset : offset + self.d_inner_local_tp]; offset += self.d_inner_local_tp
        x = zxdtAtrap[..., offset : offset + self.d_inner_local_tp]; offset += self.d_inner_local_tp
        dt = zxdtAtrap[..., offset : offset + self.nheads_local_tp]; offset += self.nheads_local_tp
        A = zxdtAtrap[..., offset : offset + self.nheads_local_tp]; offset += self.nheads_local_tp
        trap = zxdtAtrap[..., offset : offset + 2 * self.nheads_local_tp] # Trap might need 2x? check dim
        #print(f"ngroups: {self.ngroups}, mimo_dim: {self.mimo_dim}, d_state: {self.d_state}")
        B = BCangle[..., 0:self.ngroups*self.mimo_dim*self.d_state]
        C = BCangle[..., self.ngroups*self.mimo_dim*self.d_state:2*self.ngroups*self.mimo_dim*self.d_state]
        angle = BCangle[..., 2*self.ngroups*self.mimo_dim*self.d_state:] # (L, B, S)

        z = rearrange(z, "l b (G h p) -> b l (G h) p",G=self.ngroups, p=self.headdim)
        x = rearrange(x, "l b (G h p) -> b l (G h) p", G=self.ngroups, p=self.headdim)
        B = rearrange(B, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)
        dt = rearrange(dt, "l b n -> b l n").to(torch.float32)
        dt = dt.to(torch.float32)
        A = rearrange(A, "l b n -> b l n")
        trap = rearrange(trap, "l b n -> b n l")

        # value embeddings
        if self.use_ve:
            ve = self.ve_embedding(input_ids) # (B,S, G_local*D)
            x = x + self.ve_scalars.view(1, 1, -1) * ve

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)
        ADT = _A * dt

        nvtx_range_push(suffix="M3_mimo_BC_norm")
        B = self.B_norm(B)
        C = self.C_norm(C)

        if self.config.sequence_parallel:
            # All-Gather B and C and angle along Sequence dimension
            B = gather_from_sequence_parallel_region(B, group=self.pg_collection.tp)
            C = gather_from_sequence_parallel_region(C, group=self.pg_collection.tp)
            angle = gather_from_sequence_parallel_region(angle, group=self.pg_collection.tp)
        #print(f"After B/C norm and gather: B shape {B.shape}, C shape {C.shape}")
        if self.ngroups != self.nheads:
            B = B.repeat(1, 1, 1, self.n_repeat, 1) # (B, L, R, N, S)
            C = C.repeat(1, 1, 1, self.n_repeat, 1) # (B, L, R, N, S)
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
        #print(f"Before Mamba3 TileLang: C shape {C.shape}, B shape {B.shape}, x shape {x.shape}, ADT shape {ADT.shape}, dt shape {dt.shape}, trap shape {trap.shape}, angle shape {angle.shape}, D shape {self.D.shape}, z shape {z.shape}")
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
        
        #Terribles images cette norm nécessite un gather ? et l'output matrice en sequence parallelism ?
        # Sinon on utilise le trick de normalisation mathématique ?
        y = self.output_norm(y)

        return y, normed_hidden_states

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        axis_map = {
            "dt_bias": 0,
            "D": 0,
            "B_bias": 0,
            "C_bias": 0,
            "in_proj_mimo_x": 0,
            "in_proj_mimo_z": 0,
            "out_proj_mimo": 0,
            # --- FIX START ---
            # Add output_norm parameters to the map.
            # weight/bias are sharded (size d_inner_local_tp).
            #"output_norm.weight": 0,
            #"output_norm.bias": 0,
            # alpha/shift are scalars and replicated (tp_sync=True), so they are NOT sharded.
            # Removing them from axis_map ensures they are saved as singleton tensors.
            # --- FIX END ---
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

        # Splitting in_proj.weight
        #in_proj_dim = (
        #    self.d_inner_local_tp * 2
        #    + 3 * self.nheads_local_tp
        #)
        
        # Check integrity
        # assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim

        """
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.d_inner_local_tp,
                self.d_inner_local_tp,
                self.nheads_local_tp,
                self.nheads_local_tp,
                self.nheads_local_tp,
            ],
            ["z", "x", 
             "dt", "A", "trap"],
            0,
        )
        
        """
        return sharded_state_dict
    
    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)

def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )

    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim], (
            split_start,
            orig_sh_ten_no_data.local_shape[split_dim],
        )
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel(), (
            chunk_sh_tens,
            t.shape,
        )
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )