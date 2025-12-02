# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NoReturn, Optional, Tuple, Union
from contextlib import nullcontext
import math

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import (
    deprecate_inference_params,
    divide,
    get_pg_size,
    is_fa_min_version,
    is_te_min_version,
    nvtx_range_pop,
    nvtx_range_push,
)

from ..models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.transformer.enums import AttnMaskType
from .dragon_config import DragonConfig

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )

    HAVE_FA3 = True
except:
    HAVE_FA3 = False

try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    HAVE_FMLA = True
except ImportError:
    flash_mla_with_kvcache = None
    get_mla_metadata = None
    HAVE_FMLA = False

from megatron.core.transformer.transformer_config import MLATransformerConfig

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import (
        SplitAlongDim,
        TELinear,
        set_save_original_input,
    )
except ImportError:
    HAVE_TE = False
    SplitAlongDim, TELinear, set_save_original_input = None, None, None

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False

@dataclass
class SelfDiffAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_in: Union[ModuleSpec, type] = None
    linear_BkBv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

class DiffAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: SelfDiffAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        self.num_attention_heads = self.config.num_attention_heads
        self.num_signal_heads = self.config.num_signal_heads
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads

        self.query_projection_size = self.config.kv_channels * self.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "Attention pg_collection must have tp process group"
            assert hasattr(
                pg_collection, 'cp'
            ), "Attention pg_collection must have cp process group"
        self.pg_collection = pg_collection

        # Per attention head and per partition values
        world_size = get_pg_size(self.pg_collection.tp)
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, world_size)
        self.num_signal_heads_per_partition = divide(self.num_signal_heads, world_size)
        self.num_noise_heads_per_partition = divide(self.num_noise_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # To support both CUDA Graphs and key value with different hidden size
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        # Scalable softmax scalers
        if not config.intra_doc_masking:
            self.softmax_scaler = torch.nn.Parameter(torch.ones(1, 1, self.num_attention_heads_per_partition, 1))
        else:
            self.softmax_scaler = torch.nn.Parameter(torch.ones(1, self.num_attention_heads_per_partition, 1))
        setattr(self.softmax_scaler, 'tensor_model_parallel', True)

        # Diff attention scalers
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_number)
        with get_cuda_rng_tracker().fork():
        #with nullcontext(): # TEMP
            head_dim = self.hidden_size_per_attention_head // 2
            self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
            self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        for p in [self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2]:
            setattr(p, "tp_sync", True)

        self.softcap = self.config.softcap_attn

        self.core_attention1 = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            num_attention_heads=self.num_signal_heads,
            num_query_groups=self.num_signal_heads,
            cp_comm_type=cp_comm_type,
            softmax_scale=None if not self.config.use_uscaling else 1/self.key_hidden_size,
            softcap=self.softcap,
            pg_collection=self.pg_collection,
        )
        self.core_attention2 = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            num_attention_heads=self.num_noise_heads,
            num_query_groups=self.num_noise_heads,
            cp_comm_type=cp_comm_type,
            softmax_scale=None if not self.config.use_uscaling else 1/self.key_hidden_size,
            softcap=self.softcap,
            pg_collection=self.pg_collection,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        self.offload_qkv_linear = (
            self.config.fine_grained_activation_offloading
            and "qkv_linear" in self.config.offload_modules
        )

        self.offload_core_attention = (
            self.config.fine_grained_activation_offloading
            and "core_attn" in self.config.offload_modules
        )

        self.wsize_prev = None

        self.register_buffer("inv_rank", torch.tensor(1./self.config.tpa_rank), persistent=False)

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_length, batch_size, dim, dtype):
        """Allocate memory to store kv cache during inference."""

        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_query_groups_per_partition,
            dim,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _adjust_key_value_for_inference(
        self,
        inference_context: BaseInferenceContext,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rotary_pos_emb: Tensor,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Saves the generated key and value tensors to the end of the buffers in inference_context.
        Returns the full size keys and values from the provided inference_context, as well as
        adjusted rotary_pos_emb.

        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            Tuple of: query, key, value, rotary_pos_emb, attn_mask_type, block_table.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        attn_mask_type = self.attn_mask_type
        if inference_context is None:
            return query, key, value, rotary_pos_emb, attn_mask_type, None

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_context.is_static_batching():
            if self.layer_number not in inference_context.key_value_memory_dict:
                inf_max_seq_length = inference_context.max_sequence_length
                inf_max_batch_size = inference_context.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_length, inf_max_batch_size, self.key_hidden_size, key.dtype
                )
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_length, inf_max_batch_size, self.val_hidden_size, value.dtype
                )
                inference_context.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory,
                    inference_value_memory,
                )
            else:
                # Get the pre-allocated buffers for this layer
                inference_key_memory, inference_value_memory = (
                    inference_context.key_value_memory_dict[self.layer_number]
                )

        if not inference_context.is_static_batching() or inference_context.sequence_len_offset > 0:
            # This should mean that we are past the prompt forward_step
            # and so we need to turn off masking
            attn_mask_type = AttnMaskType.no_mask

        if inference_context.is_static_batching():
            batch_start = inference_context.batch_size_offset
            batch_end = batch_start + key.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_context.sequence_len_offset
            sequence_end = sequence_start + key.size(0)
            assert sequence_end <= inference_key_memory.size(0), (
                "Current sequence length is longer than expected maximum sequence length! "
                "Increase inference_max_seq_length."
            )

        if self.config.flash_decode:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None
            rotary_pos_cos_k = None
            rotary_pos_sin_k = None

            assert inference_context.is_static_batching()
            if (
                inference_context.sequence_len_offset > 0 and rotary_pos_cos is not None
            ):  # Decode phase, not prefill
                rotary_pos_cos_q = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[sequence_end - 1 : sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[sequence_end - 1 : sequence_end]
            elif rotary_pos_cos is not None:  # Prefill
                rotary_pos_cos_q = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[:sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[:sequence_end]

            # Flash Decoding assumes that the keys stored in the KV Cache already have RoPE applied.
            # Apply RoPE before we store the keys to make it compatible with flash decoding kernel
            if rotary_pos_sin_q is not None and rotary_pos_sin_k is not None:
                key = apply_rotary_pos_emb_with_cos_sin(
                    key,
                    rotary_pos_cos_k,
                    rotary_pos_sin_k,
                    rotary_interleaved=self.config.rotary_interleaved,
                )
                query = apply_rotary_pos_emb_with_cos_sin(
                    query,
                    rotary_pos_cos_q,
                    rotary_pos_sin_q,
                    rotary_interleaved=self.config.rotary_interleaved,
                )
        else:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None

        # Adjust rotary embeddings.
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if inference_context.is_static_batching():
                q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            else:
                pass
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        block_table = None
        if inference_context.is_static_batching():
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
            inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value
            key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        else:
            # Apply rotary embeddings before appending KV cache.
            if inference_context.use_flashinfer_fused_rope and (rotary_pos_cos_sin is not None):
                query, key = inference_context.apply_fused_qk_rotary_emb(
                    query, key, rotary_pos_cos_sin, self.config
                )
            elif rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                key = inference_context.apply_rotary_emb_key(
                    key, k_pos_emb, self.config, self.pg_collection.cp
                )

                rotary_pos_emb = (q_pos_emb, None)  # key rotary emb has been applied

            # Append key/value data tensors to cache.
            inference_context.append_key_value_cache(self.layer_number, key, value)

            _, max_seqlen_q = inference_context.cu_query_lengths()
            if getattr(self.config, "cache_mla_latents", None) and max_seqlen_q > 1:
                # Doing unabsorbed MLA Attention with cached mla latents (prefill/mixed mode)
                kv_cache, _, block_table = inference_context.key_value_cache(self.layer_number)
                # Uncompress the KV cache for prefill/mixed mode
                key, value = self.uncompress_kv_from_cache(kv_cache)
            else:
                # Read key/value *pointer* tensors from cache.
                key, value, block_table = inference_context.key_value_cache(self.layer_number)
        return query, key, value, rotary_pos_emb, attn_mask_type, block_table

    def _signal_noise_local_indices(self, tp_rank: int, device):
        H_tot = self.num_attention_heads
        H_local = self.num_attention_heads_per_partition
        S_tot = self.num_signal_heads
        N_tot = H_tot - S_tot
        g = math.gcd(S_tot, N_tot)
        s_block = S_tot // g
        n_block = N_tot // g
        cycle = s_block + n_block

        base = tp_rank * H_local                               # global head offset for this TP rank
        h_global = torch.arange(H_local, device=device) + base # [H_local]
        pos = h_global % cycle
        is_signal = pos < s_block
        sig_idx = torch.nonzero(is_signal, as_tuple=False).squeeze(-1) # local indices
        noi_idx = torch.nonzero(~is_signal, as_tuple=False).squeeze(-1)
        return sig_idx, noi_idx

    @abstractmethod
    def get_query_key_value_tensors(
        self, hidden_states
    ):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    @abstractmethod
    def get_gate_tensor(
        self, hidden_states
    ):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """
    
    @abstractmethod
    def normalize_qk(
        self, q, k
    ):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def flash_decode(
        self,
        sequence_len_offset: Tensor,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
        inference_key_memory: Tensor,
        inference_value_memory: Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
        rotary_interleaved: bool = False,
    ) -> (Tensor, Tensor):
        """
        The flash decoding kernel will do the following in a single execution:
        1. Compute RoPE embedding with precomputed cos & sin tensors
        2. Update the KV Cache
        3. Performs the flash attention operation
        """
        assert flash_attn_with_kvcache is not None, (
            "Flash Decoding requires the flash_attn_with_kvcache kernel, "
            "available in the flash-attn package."
        )
        q = query_layer.permute(1, 0, 2, 3)
        k = key_layer.permute(1, 0, 2, 3)
        v = value_layer.permute(1, 0, 2, 3)
        k_cache = inference_key_memory.permute(1, 0, 2, 3)
        v_cache = inference_value_memory.permute(1, 0, 2, 3)

        if rotary_cos is not None:
            rotary_cos = rotary_cos.to(query_layer.dtype)
        if rotary_sin is not None:
            rotary_sin = rotary_sin.to(query_layer.dtype)

        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=sequence_len_offset,
            rotary_interleaved=rotary_interleaved,
        )
        return out

    def flash_decode_and_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_k,
        block_table,
    ) -> Tensor:
        """Flash attention kernel for mixed decode and prefill samples.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            max_seqlen_q (int): Query total sequence length.
            max_seqlen_k (int): Key total sequence length.
            cu_seqlens_q (Tensor): Cumulative query sequence lengths.
            cu_seqlens_k (Tensor): Cumulative key sequence lengths.
            seqlens_k (Tensor): key sequence lengths.
            block_table (Tensor): KV cache block ids for all samples.
        Return:
            (Tensor) Attention output.
        """

        assert not self.training
        assert block_table is not None

        # Flash attn kernel.
        if max_seqlen_q > 1:
            q = q.squeeze(1)
            if getattr(self, "softmax_scale", None) is not None:
                softmax_scale = self.softmax_scale
            else:
                softmax_scale = q.shape[-1] ** -0.5
            if HAVE_FA3:
                # TODO(ksanthanam): Replace with call to flash_attn_varlen_func once
                # it accepts block_table
                output_total, *unused = _flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    k_new=None,
                    v_new=None,
                    qv=None,
                    out=None,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=None,
                    cu_seqlens_k_new=None,
                    seqused_q=None,
                    seqused_k=seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    page_table=block_table,
                    kv_batch_idx=None,
                    leftpad_k=None,
                    rotary_cos=None,
                    rotary_sin=None,
                    seqlens_rotary=None,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(-1, -1),
                    attention_chunk=0,
                    softcap=0.0,
                    rotary_interleaved=True,
                    scheduler_metadata=None,
                    num_splits=0,
                    pack_gqa=None,
                    sm_margin=0,
                )
            else:
                output_total = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=True,
                    block_table=block_table,
                )
            output_total = output_total.unsqueeze(1)
        else:  # decode only
            # If using MLA we use the FlashMLA kernel
            if isinstance(self.config, MLATransformerConfig):
                softmax_scale = self.softmax_scale

                num_heads_k = 1  # Only a single head for MLA Flash
                seq_len_q = 1  # Sequence length is 1 for decode
                num_heads_q = self.num_attention_heads_per_partition
                num_heads_per_head_k = seq_len_q * num_heads_q // num_heads_k

                cache_seqlens = seqlens_k
                tile_scheduler_metadata, num_splits = get_mla_metadata(
                    cache_seqlens,  # cumulative key-lengths
                    num_heads_per_head_k,  # decode-only lengths
                    num_heads_k,  # per-head dim of V
                )
                head_dim_v = self.config.kv_lora_rank
                kv_cache = k.unsqueeze(-2)
                output_total, softmax_lse = flash_mla_with_kvcache(
                    q,
                    kv_cache,
                    block_table,
                    cache_seqlens,
                    head_dim_v,
                    tile_scheduler_metadata,
                    num_splits,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            else:
                flash_attn_args = {
                    "q": q,
                    "k_cache": k,
                    "v_cache": v,
                    "cache_seqlens": seqlens_k,
                    "causal": True,
                    "page_table" if HAVE_FA3 else "block_table": block_table,
                }
                if HAVE_FA3:
                    output_total = flash_attn3_with_kvcache(**flash_attn_args)
                else:
                    output_total = flash_attn_with_kvcache(**flash_attn_args)
        return output_total

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
        window_size: Optional[Tuple[int, int]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

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
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """

        if window_size is not None and window_size[0] != self.wsize_prev and self.pg_collection.tp.rank()==0 and self.pg_collection.dp.rank()==0:
            print(f"[ATTN] Window size changed: {self.wsize_prev} -> {window_size[0]}")
            self.wsize_prev = window_size[0]

        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None
        rotary_pos_emb = None # attention layers don't have RoPE.

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                "2.7.3"
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        is_inference_mode = inference_context is not None and not self.training
        # is_using_flash_decode - True is we are using the static inference engine with flash decode
        is_using_flash_decode = is_inference_mode and self.config.flash_decode
        # is_using_flashinfer_rope - True if we are using the dynamic inference engine
        # with flashinfer fused rope
        is_using_flashinfer_rope = is_inference_mode and (
            not inference_context.is_static_batching()
            and inference_context.use_flashinfer_fused_rope
        )
        if is_using_flash_decode or is_using_flashinfer_rope:
            # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        split_qkv = True
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        if self.attention_type != "cross":
            assert not (
                self.config.fused_single_qkv_rope and split_qkv
            ), "fused_single_qkv_rope requested but not available/supported for the config."

        if self.offload_qkv_linear:
            hidden_states = fine_grained_offloading_group_start(hidden_states, name="qkv_linear")
        with get_fine_grained_offloading_context(self.offload_qkv_linear):
            qkv_output = self.get_query_key_value_tensors(hidden_states)
        if self.offload_qkv_linear:
            (qkv_output,) = fine_grained_offloading_group_commit(
                qkv_output, name="qkv_linear", forced_released_tensors=[]
            )

        attn_mask_type = self.attn_mask_type
        query, A_k, A_v, B_k, B_v, alpha_k, alpha_v, gate = qkv_output

        # q: [L, B, H_local, dk]
        # A_k: [L, B, H_local, r], A_v: [L, B, H_noise_local, r]
        # alpha_k: [L, B, H_local, 1], alpha_v: [L, B, H_noise_local, 1]
        # B_k: [L, B, r*Dk], B_v: [L, B, r*Dk]
        # gate: [L, B, H_signal_local, Dk]

        B_k = rearrange(B_k, '... (r d) -> ... r d', r=self.config.tpa_rank)
        B_v = rearrange(B_v, '... (r d) -> ... r d', r=self.config.tpa_rank)
        key = torch.matmul(A_k, B_k).mul_(self.inv_rank)
        value = torch.matmul(A_v, B_v).mul_(self.inv_rank)

        nvtx_range_pop(suffix="qkv")

        # =====================
        # kv shift
        # =====================
        if self.config.token_shift:
            key, value = self._torch_compiled_token_shift(key, value, alpha_k, alpha_v, position_ids=packed_seq_params.position_ids if packed_seq_params is not None else None)

        # =====================
        # QK norm
        # =====================
        query, key = self.normalize_qk(query, key)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            return out, None

        if (
            in_decode_mode
            and self.config.cuda_graph_impl == "local"
            and "full_iteration" not in self.config.cuda_graph_scope
            and inference_context.is_static_batching()
        ):
            raise ValueError(f"CUDA graphs must use flash decode with static batching!")

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                rotary_pos_cos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and (
            not self.config.flash_decode or inference_context is None
        ):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=_yarn_get_concentration_factor_from_config(self.config),
                    cp_group=self.pg_collection.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # scalable softmax
        # ==================================
        start_pos = 0 if inference_context is None else inference_context.sequence_len_offset
        T = query.size(0)
        wsize = window_size[0] if window_size is not None else -1
        if self.config.scalable_softmax:
            if packed_seq_params is not None:
                position_ids = packed_seq_params.position_ids+1 # (T,)
            else:
                position_ids = torch.arange(start_pos+1, start_pos+T+1, dtype=torch.long, device=query.device) # (T,)
            pos_shape = (query.size(0),) + (1,) * (query.dim() - 1)
            pos = (position_ids.to(torch.float32).view(*pos_shape))
            log_pos = pos.log() if wsize <= 0 else torch.clamp_max(pos, wsize).log()
            query = (self.softmax_scaler * log_pos) * query

        # ==================================
        # q,k splitting for diff attention
        # ==================================
        sig_idx, noi_idx = self._signal_noise_local_indices(self.pg_collection.tp.rank(), query.device)

        if packed_seq_params is None:
            index_heads = 2
            query_sig, query_noi = query.index_select(index_heads, sig_idx), query.index_select(index_heads, noi_idx)
            key_sig, key_noi = key.index_select(index_heads, sig_idx), key.index_select(index_heads, noi_idx)
            value_sig = value.repeat(1, 1, self.num_signal_heads//self.num_noise_heads, 1)
        else:
            index_heads = 1
            query_sig, query_noi = query.index_select(index_heads, sig_idx), query.index_select(index_heads, noi_idx)
            key_sig, key_noi = key.index_select(index_heads, sig_idx), key.index_select(index_heads, noi_idx)
            value_sig = value.repeat(1, self.num_signal_heads//self.num_noise_heads, 1)

        # query_sig, key_sig, value_sig: (L, B, H_signal_local, D)
        # query_noi, key_noi, value    : (L, B, H_noise_local, D)

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        if self.checkpoint_core_attention and self.training:
            core_attn_out_1 = self._checkpointed_attention_forward(
                query_sig,
                key_sig,
                value_sig,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            ) # (L, B, H_signal_local, D)
            assert len(core_attn_out_1.shape) == 4
            core_attn_out_2 = self._checkpointed_attention_forward(
                query_noi,
                key_noi,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            ) # (L, B, H_noise_local, D)
            core_attn_out_2 = core_attn_out_2.repeat(1, 1, self.num_signal_heads//self.num_noise_heads, 1) # (L, B, H_signal_local, D)
        else:
            if self.offload_core_attention and self.training:
                query = fine_grained_offloading_group_start(query, name="core_attn")
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                with get_fine_grained_offloading_context(self.offload_core_attention):
                    core_attn_out_1 = self.core_attention1(
                        query_sig.bfloat16(),
                        key_sig.bfloat16(),
                        value_sig.bfloat16(),
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        window_size=window_size,
                        concat_heads=False,
                        packed_seq_params=packed_seq_params,
                    ) # (L, B, H_signal_local, D)
                    core_attn_out_2 = self.core_attention2(
                        query_noi.bfloat16(),
                        key_noi.bfloat16(),
                        value.bfloat16(),
                        attention_mask,
                        attn_mask_type=attn_mask_type,
                        attention_bias=attention_bias,
                        window_size=window_size,
                        concat_heads=False,
                        packed_seq_params=packed_seq_params,
                    ) # (L, B, H_noise_local, D)
                    if not self.config.intra_doc_masking:
                        assert len(core_attn_out_1.shape) == 4
                        assert len(core_attn_out_2.shape) == 4
                    else:
                        assert len(core_attn_out_1.shape) == 3
                        assert len(core_attn_out_2.shape) == 3
                    assert core_attn_out_1.size(-2) == self.num_signal_heads_per_partition
                    assert core_attn_out_1.size(-1) == self.hidden_size_per_attention_head
                    assert core_attn_out_2.size(-2) == self.num_noise_heads_per_partition
                    assert core_attn_out_2.size(-1) == self.hidden_size_per_attention_head
                    if not self.config.intra_doc_masking:
                        core_attn_out_2 = core_attn_out_2.repeat(1, 1, self.num_signal_heads//self.num_noise_heads, 1) # (L, B, H_signal_local, D)
                    else:
                        core_attn_out_2 = core_attn_out_2.repeat(1, self.num_signal_heads//self.num_noise_heads, 1)
            else:
                # Dynamic batching attention kernel.
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

                core_attn_out_1 = self.flash_decode_and_prefill(
                    query_sig,
                    key_sig,
                    value_sig,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                ) # (L, B, H_signal_local, D)
                assert len(core_attn_out_1.shape) == 4
                core_attn_out_2 = self.flash_decode_and_prefill(
                    query_noi,
                    key_noi,
                    value,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    block_table,
                ) # (L, B, H_noise_local, D)
                core_attn_out_2 = core_attn_out_2.repeat(1, 1, self.num_signal_heads//self.num_noise_heads, 1) # (L, B, H_signal_local, D)

            if self.offload_core_attention and self.training:
                (core_attn_out,) = fine_grained_offloading_group_commit(
                    core_attn_out, name="core_attn", forced_released_tensors=[query, key, value]
                )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).type_as(query_sig)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).type_as(query_sig)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        core_attn_out = core_attn_out_1 - lambda_full * core_attn_out_2

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.unsqueeze(1)
        nvtx_range_pop(suffix="core_attention")

        # Output gate
        if gate is not None:
            nvtx_range_push(suffix="output_gate")
            core_attn_out = self._torch_compiled_output_gate(core_attn_out, gate)
            nvtx_range_pop(suffix="output_gate")

        return core_attn_out

    @torch.compile # TODO: reactive. it's disabled during tests
    def _torch_compiled_token_shift(self, key, value, alpha_k, alpha_v, position_ids=None):
        alpha_k = torch.sigmoid(alpha_k.float()).float().to(key.dtype) # (L, B, H_local, 1)
        alpha_v = torch.sigmoid(alpha_v.float()).float().to(value.dtype) # (L, B, H_noise_local, 1)
        if position_ids is not None:
            # first token of each doc has pos==0
            doc_start = (position_ids == 0).view(-1, 1, 1, 1) # (L, 1, 1, 1) bool
        else:
            L, B = key.shape[:2]
            doc_start = torch.zeros(L, 1, 1, 1, dtype=torch.bool, device=key.device)
            doc_start[0] = True

        alpha_k = alpha_k.masked_fill(doc_start, 0)
        alpha_v = alpha_v.masked_fill(doc_start, 0)

        key_prev = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, 0, 1, 0))[:-1] # (L, B, H_local, D)
        value_prev = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, 0, 1, 0))[:-1] # (L,B, H_noise_local, D)

        key = alpha_k * key_prev + (1 - alpha_k) * key
        value = alpha_v * value_prev + (1 - alpha_v) * value

        return key, value

    @torch.compile # TODO: reactive. it's disabled during tests
    def _torch_compiled_output_gate(self, x, gate):
        x_dtype = x.dtype
        gate = gate.contiguous().view(*x.shape)
        x = x * torch.nn.functional.silu(gate.float() + 1.15)
        x = x.to(x_dtype)
        return x

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        raise NotImplementedError("set_for_recompute_input_layernorm is not implemented.")

class SelfDiffAttention(DiffAttention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: SelfDiffAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        Dk = self.config.kv_channels
        r = self.config.tpa_rank
        alpha_dim = 1 if self.config.token_shift else 0
        gate_dim = Dk if self.config.gate_attn else 0

        # we make groups of 4 heads (q, A_k, alpha_k), 1 noise head (A_v, alpha_v), and 3 signal heads (gate).
        # we call each group a "super head".
        self.num_super_heads = self.config.num_attention_heads // 4
        self.num_super_heads_per_partition = self.num_super_heads // self.pg_collection.tp.size()
        self.super_head_dim = 4 * (Dk + r + alpha_dim) + 1 * (r + alpha_dim) + 3 * (gate_dim)
        out_dim = self.num_super_heads * self.super_head_dim
        self.linear_in = build_module(
            submodules.linear_in,
            self.config.hidden_size,
            out_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            return_layernorm_output=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='in',
            tp_group=self.pg_collection.tp,
        )

        out_dim = 2 * r * Dk
        self.linear_BkBv = build_module(
            submodules.linear_BkBv,
            self.config.hidden_size,
            out_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode='duplicated',
            is_expert=False,
            tp_comm_buffer_name='BkBv',
        )
        setattr(self.linear_BkBv.weight, 'tp_sync', True)

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during
        data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably
        not required; transmitting hashes is sufficient."""

        if not self.config.qk_layernorm:
            return

        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

        def _compare(srcs, tgts, names, parallelism):
            assert len(srcs) == len(tgts) == len(names)
            for src, tgt, name in zip(srcs, tgts, names):
                assert torch.all(src == tgt), (
                    f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. "
                    f"Diff: {torch.norm(src - tgt)}"
                )

        for i, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )

        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

        for i, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )

    def get_query_key_value_tensors(self, hidden_states):
        Dk = self.config.kv_channels
        r = self.config.tpa_rank
        alpha_dim = 1 if self.config.token_shift else 0
        gate_dim = Dk if self.config.gate_attn else 0

        out, _ = self.linear_in(hidden_states) # [L, B, super_head_dim * num_super_heads]
        mixed_in, normed_hidden_states = out
        mixed_in = rearrange(mixed_in, "l b (H p) -> l b H p", H=self.num_super_heads_per_partition)#.contiguous()
        # split per super head: [L, B, H_super_local, super_head_dim]
        mixed_heads = mixed_in[..., 0:4*(Dk + r + alpha_dim)]; accum = 4*(Dk + r + alpha_dim)
        mixed_noise_heads = mixed_in[..., accum:accum + 1*(r + alpha_dim)]; accum += 1*(r + alpha_dim)
        mixed_signal_heads = mixed_in[..., accum:accum+3*Dk]

        # mixed_heads: [L, B, H_super_local, 4*(Dk + r + alpha_dim)]
        # mixed_noise_heads: [L, B, H_super_local, 1*(r + alpha_dim)]
        # mixed_signal_heads: [L, B, H_super_local, 3*Dk]

        # split mixed_heads into q, A_k, (alpha_k)
        mixed_heads = rearrange(mixed_heads, "l b h (n d) -> l b (h n) d", n=4) # [L, B, H_local, D]
        q = mixed_heads[..., 0:Dk]; accum = Dk
        A_k = mixed_heads[..., accum:accum+r]; accum += r
        alpha_k = mixed_heads[..., accum:accum+alpha_dim] if self.config.token_shift else None
        # split mixed_noise_heads into A_v, (alpha_v)
        mixed_noise_heads = rearrange(mixed_noise_heads, "l b h (n d) -> l b (h n) d", n=1) # [L, B, H_noise_local, D]
        A_v = mixed_noise_heads[..., 0:r]; accum = r
        alpha_v = mixed_noise_heads[..., accum:] if self.config.token_shift else None
        # split mixed_signal_heads into gate
        mixed_signal_heads = rearrange(mixed_signal_heads, "l b h (n d) -> l b (h n) d", n=3) # [L, B, H_signal_local, D]
        gate = mixed_signal_heads[..., 0:gate_dim] if self.config.gate_attn else None

        # project to B_k, B_v (shared across TP ranks)
        mixed_BkBv, _ = self.linear_BkBv(normed_hidden_states) # [L, B, 2*r*Dk]
        B_k, B_v = torch.split(mixed_BkBv, r * Dk, dim=-1) # [L, B, r*Dk], [L, B, r*Dk]

        # q: [L, B, H_local, dk], A_k: [L, B, H_local, r], alpha_k: [L, B, H_local, 1]
        # A_v: [L, B, H_noise_local, r], alpha_v: [L, B, H_noise_local, 1]
        # gate: [L, B, H_signal_local, Dk]
        # B_k: [L, B, r*Dk], B_v: [L, B, r*Dk]

        if self.config.test_mode:
            self.run_realtime_tests()

        return q, A_k, A_v, B_k, B_v, alpha_k, alpha_v, gate

    def get_gate_tensor(self, hidden_states):
        Dk = self.config.kv_channels
        mixed_g, _ = self.linear_g(hidden_states) # [L, B, H_signal_local * Dk]
        mixed_g = rearrange(mixed_g, "l b (h d) -> l b h d", h=self.num_signal_heads_per_partition)#.contiguous()
        # split per head: [L, B, H_signal_local, Dk]
        g = mixed_g[..., 0:Dk]
        return g

    def normalize_qk(self, q, k):
        if self.q_layernorm is not None:
            q = self.q_layernorm(q)

        if self.k_layernorm is not None:
            k = self.k_layernorm(k)
        return q, k

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations"""
        self._backward_proj()
        self._backward_BkBv()

    def _backward_proj(self):
        """Update weights for QKV projection layer"""
        self.linear_in.backward_dw()
    
    def _backward_BkBv(self):
        """Update weights for BkBv projection layer"""
        self.linear_BkBv.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        from megatron.core.extensions.transformer_engine import set_save_original_input

        set_save_original_input(self.linear_in)
