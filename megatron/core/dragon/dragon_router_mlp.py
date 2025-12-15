# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import gc
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import math
import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.activations import fast_gelu

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


logger = logging.getLogger(__name__)

# TODO : need to make sure these are synced across the TP ranks!!!!!!!


class DragonRouterMLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: DragonConfig,
    ):
        super().__init__(config=config)

        self.config: DragonConfig = config

        self.input_size = self.config.hidden_size
        self.intermediate_size = self.input_size//8
        self.output_size = self.config.num_moe_experts

        self.linear_down = TELinear(
            self.input_size,
            self.intermediate_size,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=True,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="down",
        )

        self.eda_scalers = torch.nn.Parameter(torch.zeros(self.intermediate_size))

        self.eda_norm = TENorm(
            config=self.config,
            hidden_size=self.intermediate_size,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = TELinear(
            self.intermediate_size,
            self.intermediate_size,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=True,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="fc1",
        )

        self.linear_fc2 = TELinear(
            self.intermediate_size,
            self.intermediate_size,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=True,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="fc2",
        )

        self.linear_fc3 = TELinear(
            self.intermediate_size,
            self.output_size,
            config=self.config,
            parallel_mode="duplicated",
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name="fc3",
        )

        if self.config.use_te_activation_func:
            self.activation_func = transformer_engine.pytorch.ops.GELU()
        else:
            self.activation_func = F.gelu

    def forward(self, hidden_states, prev_hs=None, per_token_scale=None):
        nvtx_range_push(suffix="linear_down")
        intermediate_parallel, bias_parallel = self.linear_down(hidden_states)
        nvtx_range_pop(suffix="linear_down")

        intermediate_parallel = self._torch_compiled_EDA(intermediate_parallel, bias_parallel, self.eda_scalers, prev_hs)
        stashed_hs = intermediate_parallel.clone()
        intermediate_parallel = self.eda_norm(intermediate_parallel)

        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation1")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation1")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        intermediate_parallel, bias_parallel = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        nvtx_range_push(suffix="activation2")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation2")

        nvtx_range_push(suffix="linear_fc3")
        output, output_bias = self.linear_fc3(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc3")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, stashed_hs
    
    @torch.compile
    def _torch_compiled_EDA(self, x, bias, lambd, x_prev=None):
        if x_prev is not None:
            return x + bias + lambd * x_prev
        else:
            return x + bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            if name == "eda_norm" or name == "activation_func":
                continue
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key
        if flattened_range is None:
            tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
            return [
                ShardedTensor.from_rank_offsets(
                    w_key,
                    tensor_w,
                    *sharded_offsets,
                    offset_w,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
                ShardedTensor.from_rank_offsets(
                    v_key,
                    tensor_v,
                    *sharded_offsets,
                    offset_v,
                    replica_id=replica_id,
                    prepend_axis_num=prepend_axis_num,
                ),
            ]
        else:
            if singleton_local_shards:
                raise NotImplementedError(
                    'singleton_local_shards not implemented for SwiGLU MLP flattened tensors'
                )
            # Here we need to map a slice `t` (`flattened_range` specifies slice start and stop)
            # of the *original* flattened tensor into slices `w` and `v` of chunked
            # and flattened tensor.
            # Example:
            # If original tensor has (16, 5) shape and flattened_range is `slice(8, 64)`,
            # then `t` has shape `(56,)` and we need to create 2 tensors:
            # w: first 32 elements of `t` with flattened_range slice(8, 40)
            # v: last 24 elements of `t` with flattened_range slice(0, 24)
            # Global offsets are the same as in the non-flattened case
            assert t.ndim == 1, (key, t.shape)
            non_flat_local_shape = (original_shape[0] // 2, *original_shape[1:])
            chunk_numel = original_numel // 2
            result = []
            if flattened_range.start < chunk_numel:
                # Non-empty `w` chunk
                tensor_w = t[: chunk_numel - flattened_range.start]
                flattened_range_w = slice(
                    flattened_range.start, min(chunk_numel, flattened_range.stop)
                )
                assert len(tensor_w) == flattened_range_w.stop - flattened_range_w.start
                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_w,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_w,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_w,
                    )
                )
            if flattened_range.stop > chunk_numel:
                # Non-empty `v` chunk
                tensor_v = t[-(flattened_range.stop - chunk_numel) :]
                flattened_range_v = slice(
                    max(chunk_numel, flattened_range.start) - chunk_numel,
                    flattened_range.stop - chunk_numel,
                )
                assert len(tensor_v) == flattened_range_v.stop - flattened_range_v.start, (
                    len(tensor_v),
                    flattened_range_v,
                )

                result.append(
                    ShardedTensor.from_rank_offsets_flat(
                        key,
                        tensor_v,
                        non_flat_local_shape,
                        *sharded_offsets,
                        offset_v,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                        flattened_range=flattened_range_v,
                    )
                )
            assert sum(sh_ten.data.numel() for sh_ten in result) == t.numel(), (result, t.shape)
            return result

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            try:
                return torch.cat(sub_state_dict)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(
                    f"CUDA OutOfMemoryError encountered during tensors merging."
                    f" Switching to CPU merge. (Error: {e})"
                )
                merged_sub_state_dict = torch.cat([t.cpu() for t in sub_state_dict])
                gc.collect()
                torch.cuda.empty_cache()
                return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
