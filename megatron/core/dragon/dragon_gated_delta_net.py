# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union
from einops import rearrange
from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push, make_tp_sharded_tensor_for_checkpoint

from .dragon_config import DragonConfig

# TODO : state passing

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.utils import prepare_sequence_ids

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule, prepare_sequence_ids = None, None

    HAVE_FLA = False

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


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

@dataclass
class GatedDeltaNetSubmodules:
    """
    Contains the module specs for the input linear, output norm, and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = IdentityOp

class GatedDeltaNet(MegatronModule):
    """Gated Delta Net (GDN) layer class

    GDN layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        input_scalar: float = 1.,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Args:
            config: The config of the model.
            submodules: Contains the module specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization in the kernel of the gated delta rule.
            A_init_range: The initialization range for the attention weights.
            pg_collection: The required process groups to use for tensor model parallel and context
                parallel.
        """

        if not HAVE_FLA:
            raise ImportError("FLA is not installed. Please install it with `pip install fla`.")

        super().__init__(config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.config = config
        self.hidden_size = config.hidden_size
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        assert self.num_key_heads == self.num_value_heads, "MVA not supported."
        self.num_heads = self.num_key_heads
        self.num_heads_local = self.num_heads // self.tp_size

        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the innermost dimension of the GDN layer "
                "input projection output tensor must be a multiple of 16."
            )
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
            alpha_fwd=input_scalar,
            alpha_bwd=input_scalar,
        )

        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        # weight shape: [conv_dim, 1, d_conv]
        # bias shape: [conv_dim]
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        setattr(self.conv1d.weight, "tensor_model_parallel", True)
        if conv_bias:
            setattr(self.conv1d.bias, "tensor_model_parallel", True)

        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        # dt_bias parameter
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        # A_log parameter
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
            #with nullcontext():
                # conv1d.weight
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                # dt_bias
                dt_min = 0.001
                dt_max = 0.1
                dt_init_floor = 1e-4
                dt = torch.exp(
                    torch.rand(self.num_heads_local) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                )
                dt = torch.clamp(dt, min=dt_init_floor)
                # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                with torch.no_grad():
                    self.dt_bias.data.copy_(inv_dt)
                # A_log
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                with torch.no_grad():
                    self.A_log.data.copy_(torch.log(A))
                with torch.no_grad():
                    self.conv1d.weight.normal_(0, self.config.init_method_std)

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
        Perform a forward pass through the GDN module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) GDN output and bias.

        """
        # TODO: Deal with attention_mask

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size

        if inference_context is not None:
            assert (
                inference_context.is_static_batching()
            ), "GDN does not currently support dynamic inference batching."
            assert not self.config.sequence_parallel
            # TODO: support inference
            raise NotImplementedError("GDN does not support inference for now.")

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")
        qkvzba = qkvzba.transpose(0, 1) # s b x --> b s x
        qkvzba = rearrange(qkvzba, "b l (h p) -> b l h p", h=self.num_heads_local)#.contiguous()
        # split per head: [L, B, H_local, dk+dk+dv/dv/1/1] where dq=dk=do
        qkv = qkvzba[..., :2*self.key_head_dim+self.value_head_dim]; accum = 2*self.key_head_dim+self.value_head_dim
        gate = qkvzba[..., accum:accum+self.value_head_dim]; accum += self.value_head_dim
        beta = qkvzba[..., accum:accum+1].squeeze(-1); accum += 1
        alpha = qkvzba[..., accum:accum+1].squeeze(-1)

        # qkv: (B, L, H_local, D)
        # gate: (B, L, H_local, Dv)
        # beta: (B, L, H_local)
        # alpha: (B, L, H_local)

        # Convolution on qkv
        qkv = rearrange(qkv, 'b l h d -> b l (h d)')
        qkv = qkv.transpose(1, 2)
        nvtx_range_push(suffix="conv1d")
        if causal_conv1d_fn is None:
            qkv = F.silu(self.conv1d(qkv)[..., :seq_len])
        else:
            seq_idx = None
            if packed_seq_params is not None:
                seq_idx = prepare_sequence_ids(packed_seq_params.cu_seqlens_q).to(torch.int32).unsqueeze(0)
            qkv = causal_conv1d_fn(
                x=qkv,
                weight=self.conv1d.weight.squeeze(1), # d, 1, w -> d, w
                bias=self.conv1d.bias,
                activation='silu',
                seq_idx=seq_idx,
            )
        nvtx_range_pop(suffix="conv1d")
        # Split qkv into query, key, and value
        qkv = qkv.transpose(1, 2) # b, d, s -> b, s, d
        qkv = rearrange(qkv, "b l (h p) -> b l h p", h=self.num_heads_local)#.contiguous()
        query = qkv[..., :self.key_head_dim]; accum = self.key_head_dim
        key = qkv[..., accum:accum+self.key_head_dim]; accum += self.key_head_dim
        value = qkv[..., accum:accum+self.value_head_dim]

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)  # In fp32
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query.bfloat16(),
            key.bfloat16(),
            value.bfloat16(),
            g=g,
            beta=beta,
            scale=None if not (self.config.use_uscaling or self.config.use_completedp) else 1/self.key_head_dim,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm,
            cu_seqlens=packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        # Output gate
        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._torch_compiled_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        core_attn_out = core_attn_out.transpose(0, 1).contiguous()  # b s h d -> s b h d

        return core_attn_out, 0

    @torch.compile
    def _torch_compiled_output_gate(self, x, gate):
        x_dtype = x.dtype
        gate = gate.contiguous().view(*x.shape)
        x = x * F.silu(gate.float() + 1.15)
        x = x.to(x_dtype)
        return x
    
    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        axis_map = {
            'A_log': 0,
            'dt_bias': 0,
        }

        # Parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map=axis_map, # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            if 'conv1d' in name:
                # Add TP sharding for Conv1d
                module_sd = module.state_dict(prefix='', keep_vars=True)
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd, f'{prefix}{name}.', {f'weight': 0, f'bias': 0}, sharded_offsets
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata
                )

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict
