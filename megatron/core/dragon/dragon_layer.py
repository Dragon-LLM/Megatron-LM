# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.distributed
from torch import Tensor
import torch.nn as nn
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_commit,
    fine_grained_offloading_group_start,
    get_fine_grained_offloading_context,
)
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

from .dragon_ddl import ResidualShortConvCompressor, DeepDeltaResidualExpanded

logger = logging.getLogger(__name__)


def get_dragon_layer_offset(
    config: DragonConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0

    if config.pipeline_model_parallel_size > 1:

        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Dragon
            # layers after deducting the Dragon layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            middle_pipeline_rank = (
                pp_rank if config.num_layers_in_first_pipeline_stage is None else pp_rank - 1
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pp_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + middle_pipeline_rank
                        * (
                            num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                if pp_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            # import here to avoid circular import
            from megatron.core.pipeline_parallel.utils import is_vp_first_stage

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pp_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
                ):
                    offset -= 1
            else:
                offset = pp_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
                ):
                    offset -= 1
    else:
        offset = 0
    return offset

class DragonGeodesicNorm(nn.Module):
    def __init__(self, config: DragonConfig, layer_idx: int):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))
        self.clamp = torch.pi/4
        self.register_buffer("layer_idx", torch.tensor(layer_idx), persistent=False)
        if config.sequence_parallel:
            setattr(self.scale, 'tp_sync', True)
            setattr(self.bias, 'tp_sync', True)

    @torch.compile
    def forward(self, x, g):
        """
        x: residual;
        g: ffn(x) or attn(x);
        """

        gradient = g - (x * g).sum(dim=-1,keepdim=True) / (torch.norm(x, p=2, dim=-1, keepdim=True) ** 2) * x
        tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True)
        safe_tangent_norm = torch.clamp(tangent_norm, min=1e-8)
        unit_tangent = gradient / safe_tangent_norm
        R = torch.norm(x, p=2, dim=-1, keepdim=True)
        safe_R = torch.clamp(R, min=1e-6)
        theta = torch.clamp(safe_tangent_norm / safe_R, max=self.clamp)
        theta = torch.clamp((theta * self.scale + self.bias) / self.layer_idx, max=self.clamp)
        output = x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)
        return output
    
@dataclass
class DragonLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Dragon layer.

    This class defines the structure and default implementations for various
    components of a Dragon layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    attention: Union[ModuleSpec, type] = IdentityOp
    attention_v2: Union[ModuleSpec, type] = IdentityOp
    gdn: Union[ModuleSpec, type] = IdentityOp
    mamba3: Union[ModuleSpec, type] = IdentityOp
    mixer_norm: Union[ModuleSpec, type] = IdentityFuncOp
    mixer_proj: Union[ModuleSpec, type] = IdentityOp
    pre_mlp_norm: Union[ModuleSpec, type] = IdentityFuncOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    moe: Union[ModuleSpec, type] = IdentityOp
    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseDragonLayer(ABC):
    """A common parent class for `DragonLayer` like implementations.

    A dummy class that is subclassed by similar `DragonLayer`s e.g. the
    `DragonLayer` in this file and possibly other `DragonLayer`
    implementations that aim to use `DragonBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `DragonBlock`. See `_get_block_submodules` method
    implementation in `dragon_block.py` file for more details.
    """

    def __init__(self):
        pass


class DragonLayer(GraphableMegatronModule, BaseDragonLayer):
    """A single Dragon layer.

    Dragon layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: DragonConfig,
        submodules: DragonLayerSubmodules,
        layer_mixer_type: str,
        layer_mlp_type: str,
        layer_number: int = 1,
        vocab_size: int = 50000,
        use_ve: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config, vp_stage=vp_stage)
        #global CONFIG_MLP
        #if CONFIG_MLP is None:
        CONFIG_MLP = config
            #CONFIG_MLP.fp8 = "hybrid"
            
            
        #print("INIT of layer", layer_mixer_type, layer_number)
        
        #self.mlp_duration = 0.0
        #self.attention_duration = 0.0
        #self.nb_forward = 0
        #self.start_of_layer = torch.cuda.Event(enable_timing=True)
        #self.end_of_attention = torch.cuda.Event(enable_timing=True)
        #self.end_of_mlp = torch.cuda.Event(enable_timing=True)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.submodules_config = submodules
        layer_offset = get_dragon_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.layer_number = layer_number + layer_offset
        #print("Layer number : ", layer_number, " | Layer Offset: ", layer_offset, "| Adjusted Layer number : ", self.layer_number, " | VP stage: ", vp_stage)

        lns = 1.
        if self.config.use_lns:
            lns = self.layer_number ** (-0.5)

        # [Module 1: Mixer]
        if layer_mixer_type == 'T' or layer_mixer_type == 'V':
            attention_optional_kwargs = {}
            if config.context_parallel_size > 1 and config.cp_comm_type is not None:
                if isinstance(config.cp_comm_type, list):
                    attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
                else:
                    attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

            attention_optional_kwargs["pg_collection"] = pg_collection

            self.mixer = build_module(
                submodules.attention if layer_mixer_type == 'T' else submodules.attention_v2,
                config=self.config,
                layer_number=self.layer_number,
                vocab_size=vocab_size,
                use_ve=use_ve,
                input_scalar=lns,
                **attention_optional_kwargs,
            )
            num_mixer_heads = self.mixer.num_signal_heads
            num_mixer_heads_local = self.mixer.num_signal_heads_per_partition
            head_dim = self.mixer.val_hidden_size
        elif layer_mixer_type == 'g':
            self.mixer = build_module(
                submodules.gdn,
                config=self.config,
                layer_number=self.layer_number,
                vocab_size=vocab_size,
                use_ve=use_ve,
                input_scalar=lns,
                pg_collection=pg_collection,
            )
            num_mixer_heads = self.mixer.num_heads
            num_mixer_heads_local = self.mixer.num_heads_local
            head_dim = self.mixer.value_head_dim
        elif layer_mixer_type == 'M':
            self.mixer = build_module(
                submodules.mamba3,
                config=self.config,
                layer_number=self.layer_number,
                vocab_size=vocab_size,
                use_ve=use_ve,
                input_scalar=lns,
                pg_collection=pg_collection,
            )
            num_mixer_heads = self.mixer.nheads
            num_mixer_heads_local = self.mixer.nheads_local_tp
            head_dim = self.mixer.headdim
        else:
            raise ValueError(f"Unsupported layer mixer type: {layer_mixer_type}")

        # [Module 2: Mixer norm]
        if config.mixer_gn:
            self.mixer_norm = build_module(
                submodules.mixer_norm,
                config=self.config,
                hidden_size=head_dim,
                eps=self.config.layernorm_epsilon,
                use_weights=False, # manual scalers
            )
            if not config.layernorm_zero_centered_gamma:
                self.mixer_norm_scalers = torch.nn.Parameter(torch.ones(1, 1, num_mixer_heads_local, head_dim))
            else:
                self.mixer_norm_scalers = torch.nn.Parameter(torch.zeros(1, 1, num_mixer_heads_local, head_dim))

        # [Module 3: Mixer projection]
        self.mixer_proj = build_module(
            submodules.mixer_proj,
            num_mixer_heads*head_dim,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            input_is_parallel=True,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='mixer_proj',
            tp_group=self.pg_collection.tp,
        )

        self.pre_mlp_norm = build_module(
            submodules.pre_mlp_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 4: MLP block]
        if layer_mlp_type == 'd':
            self.mlp = build_module(
                submodules.mlp,
                config=self.config,
                input_scalar=lns,
                tp_group=self.pg_collection.tp
            )
        elif layer_mlp_type == 'm':
            self.mlp = build_module(
                submodules.moe,
                config=self.config,
                input_scalar=lns,
                pg_collection=pg_collection,
            )

        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)
        self.is_moe_layer = layer_mlp_type == 'm'

        self.recompute_mlp = False
        self.recompute_pre_mlp_layernorm = False

        if self.config.recompute_granularity == 'selective':
            if "mlp" in self.config.recompute_modules:
                if not self.is_moe_layer:
                    self.recompute_mlp = True
            if "layernorm" in self.config.recompute_modules:
                self.recompute_pre_mlp_layernorm = True
                if self.config.fp8:
                    if isinstance(self.mlp, MoELayer):
                        self.mlp.set_for_recompute_pre_mlp_layernorm()
                    else:
                        from megatron.core.extensions.transformer_engine import set_save_original_input
                        set_save_original_input(self.mlp.linear_fc1)

        self.offload_mixer_proj = (
            self.config.fine_grained_activation_offloading
            and "mixer_proj" in self.config.offload_modules
        )
        self.offload_mlp_norm = False
        
        if self.config.use_ddl:
            self.compress = ResidualShortConvCompressor(self.config)
            # Store compiled callables separately to avoid pickle errors during
            # distributed checkpointing (torch.compile wraps produce unpicklable
            # code objects). The original nn.Module stays registered as a child
            # for proper state_dict / sharded_state_dict handling.
            self._compiled_compress = torch.compile(self.compress)
            self.ddl_mixer = DeepDeltaResidualExpanded(self.config, input_scalar=lns)
            self._compiled_ddl_mixer = torch.compile(self.ddl_mixer)
            self.ddl_mlp = DeepDeltaResidualExpanded(self.config, input_scalar=lns)
            self._compiled_ddl_mlp = torch.compile(self.ddl_mlp)

        if self.config.use_geodesic_norm:
            self.geodesic_mixer = DragonGeodesicNorm(self.config, self.layer_number)
            self.geodesic_mlp = DragonGeodesicNorm(self.config, self.layer_number)

        a, b = 1., 1.
        if self.config.use_uscaling:
            a = self.config.uscaling_tau ** (0.5)
            b = (1. - self.config.uscaling_tau) ** (0.5)
        elif self.config.use_completedp:
            a = (len(self.config.layers_mixer_config)/len(self.config.layers_mixer_config_base)) ** (-self.config.completedp_alpha)
        self.register_buffer("a", torch.tensor(a), persistent=False)
        self.register_buffer("b", torch.tensor(b), persistent=False)

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: DragonConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "DragonLayer._get_layer_offset is deprecated."
            "Please use get_dragon_layer_offset instead."
        )
        return get_dragon_layer_offset(config)

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
        #self.start_of_layer.record()
        residual, x_in, hidden_states, y_mixer, stashed_hs = self._forward_mixer(*args, **kwargs) # (L, B, H, D)
        #print("After mixer forward: ", hidden_states.shape, " y_mixer shape: ", y_mixer.shape)
        if self.config.mixer_gn:
            y_mixer = self._torch_compiled_headwise_norm(y_mixer)
        y_mixer = y_mixer.view(y_mixer.size(0), y_mixer.size(1), -1) # (L, B, H*D)
        nvtx_range_push(suffix="mixer_proj")
        if self.offload_mixer_proj:
            y_mixer = fine_grained_offloading_group_start(y_mixer, name="mixer_proj")
        with get_fine_grained_offloading_context(self.offload_mixer_proj):
            y_mixer, _ = self.mixer_proj(y_mixer)
        if self.offload_mixer_proj:
            y_mixer, _ = fine_grained_offloading_group_commit(
                y_mixer, None, name="mixer_proj", forced_released_tensors=[y_mixer]
            )
        nvtx_range_pop(suffix="mixer_proj")
        # print("Before mixer residual write: ", residual.shape, " y_mixer shape: ", y_mixer.shape)
        if self.config.use_geodesic_norm:
            residual = self.geodesic_mixer(residual, y_mixer)
        elif not self.config.use_ddl:
            residual = self._torch_compiled_residual_write(residual, y_mixer, self.b, self.a)
        else:
            residual = self._compiled_ddl_mixer(residual, k_in=y_mixer, v_in=x_in, context=hidden_states, scalar=self.a)
            #print("After mixer residual write: ", residual.shape)
            #self.end_of_attention.record()
            # print("After mixer residual write: ", residual.shape)
        
        hidden_states = self.pre_mlp_norm(residual)

        # When fp8_mlp_only is enabled, the block-level FP8 context is disabled,
        # so we apply it here to wrap only the MLP/MoE computation.
        if self.config.fp8 and self.config.fp8_mlp_only:
            from contextlib import nullcontext
            from megatron.core.fp8_utils import is_first_last_bf16_layer
            if is_first_last_bf16_layer(self.config, self.layer_number - 1):
                mlp_fp8_context = nullcontext()
            else:
                mlp_fp8_context = get_fp8_context(self.config, self.layer_number - 1)
        else:
            from contextlib import nullcontext
            mlp_fp8_context = nullcontext()

        with mlp_fp8_context:
            x_in, hidden_states, out = self._forward_mlp(residual, hidden_states, stashed_hs, kwargs.get("inference_context", None))
        stashed_hs = None
        if self.is_moe_layer:
            y_mlp = out[0]
            stashed_hs = out[4]
        else:
            y_mlp = out[0]
        #self.end_of_mlp.record()
        #torch.cuda.synchronize()
        """
        if self.nb_forward != 0:
            self.attention_duration += self.start_of_layer.elapsed_time(self.end_of_attention)
            self.mlp_duration += self.end_of_attention.elapsed_time(self.end_of_mlp)
        self.nb_forward += 1
        if self.nb_forward % 100 == 0:
            log_single_rank(
                logger,
                logging.INFO,
                f"Layer {self.layer_number} average attention time: "
                f"{self.attention_duration / (self.nb_forward-1)} ms, "
                f"average mlp time: {self.mlp_duration / (self.nb_forward-1)} ms",
            )
        #"""
        if self.config.use_geodesic_norm:
            residual = self.geodesic_mlp(residual, y_mlp)
        elif not self.config.use_ddl:
            residual = self._torch_compiled_residual_write(residual, y_mlp, self.b, self.a)
        else:
            residual = self._compiled_ddl_mlp(residual, k_in=y_mlp, v_in=x_in, context=hidden_states, scalar=self.a)
        #print("After final residual write: ", residual.shape, flush=True)
        return residual, stashed_hs

    def _forward_mixer(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        window_size: Optional[Tuple[int, int]] = None,
        input_ids: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        stashed_hs: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_group_commit,
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states
                
        # compress
        x_in = residual
        if self.config.use_ddl:
            x_in = self._compiled_compress(residual)

        # Self attention.
        nvtx_range_push(suffix="mixer")
        y_mixer, normed_hidden_states = self.mixer(
            x_in,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            window_size=window_size,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="mixer")

        return residual, x_in, normed_hidden_states, y_mixer, stashed_hs

    def _forward_mlp(self, residual, hidden_states, stashed_hs, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            fine_grained_offloading_group_start,
            get_fine_grained_offloading_context,
        )

        # compress
        x_in = residual
        if self.config.use_ddl:
            x_in = self._compiled_compress(residual)

        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    self.pre_mlp_norm, x_in
                )
        else:
            with get_fine_grained_offloading_context(self.offload_mlp_norm):
                pre_mlp_layernorm_output = self.pre_mlp_norm(x_in)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )

        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and 'moe_router' in self.config.cuda_graph_scope
        ):
            cudagraph_outputs = self.mlp(pre_mlp_layernorm_output)
            return cudagraph_outputs + [residual]
        elif self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, stashed_hs)
            #print("MLP output with bias type: ", mlp_output_with_bias)
            #output, mlp_bias, routing_map, stashed_hs
            #print("MLP output : ", mlp_output, " | Bias: ", bias)
            #mlp_output_with_bias = (mlp_output.to(torch.bfloat16), bias)
            
        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        nvtx_range_pop(suffix="mlp")

        return x_in, pre_mlp_layernorm_output, mlp_output_with_bias

    @torch.compile
    def _torch_compiled_headwise_norm(self, y_mixer):
        if not self.config.layernorm_zero_centered_gamma:
            y_mixer = self.mixer_norm(y_mixer) * self.mixer_norm_scalers
        else:
            y_mixer = self.mixer_norm(y_mixer) * (self.mixer_norm_scalers + 1.)
        return y_mixer
    
    @torch.compile
    def _torch_compiled_residual_write(self, residual, y_mixer, a, b):
        residual = a * residual + b * y_mixer
        return residual

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        tensor_parallel_layers_axis_map = None
        if self.config.mixer_gn:
            tensor_parallel_layers_axis_map={
                'mixer_norm_scalers': 2,
            }
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata, tensor_parallel_layers_axis_map)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the transformer layer. Besides the hidden_states that is
        generated in GraphableMegatronModule, we also add the attention_mask.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        if not isinstance(self.mixer, IdentityOp) and (
            not self.config.cuda_graph_scope or 'mixer' in self.config.cuda_graph_scope
        ):
            slen_per_cp = seq_length // self.config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((slen_per_cp, seq_length))).bool())
                .to(torch.cuda.current_device())
                .reshape(1, 1, slen_per_cp, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )
        return static_inputs

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs.
        """
        if not self.config.cuda_graph_scope:
            return super()._get_submodules_under_cudagraphs()

        submodules = []
        if 'mixer' in self.config.cuda_graph_scope:
            submodules += [
                self.attention,
            ]
        if (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope) or (
            self.is_moe_layer and 'moe' in self.config.cuda_graph_scope
        ):
            submodules += [self.mlp]
        elif self.is_moe_layer and 'moe_router' in self.config.cuda_graph_scope:
            submodules += [self.mlp.router]
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                submodules += [self.mlp.shared_experts]
        return submodules

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """
        CUDA Graph capture for this layer using TE interface.
        There are some differences from the normal pass:
        1. In some conditions CUDA graph cannot cover the entire layer. The `cuda_graph_scope`
           attribute can be set to control the scope of the CUDA graph.
        2. If context is None, it cannot be returned as output.
        """
        context = None
        if not self.config.cuda_graph_scope or 'mixer' in self.config.cuda_graph_scope:
            hidden_states, context = self._forward_mixer(*args, **kwargs)
        else:
            if len(args) > 0:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.pop("hidden_states")

        if (
            not self.config.cuda_graph_scope
            or (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope)
            or (
                self.is_moe_layer
                and (
                    'moe' in self.config.cuda_graph_scope
                    or 'moe_router' in self.config.cuda_graph_scope
                )
            )
        ):
            hidden_states = self._forward_mlp(hidden_states)
        if not isinstance(hidden_states, list) and not isinstance(hidden_states, tuple):
            cuda_graph_outputs = [hidden_states]
        else:
            cuda_graph_outputs = list(hidden_states)
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, `inference_context` and `packed_seq_params` are excluded from input list.
        """
        context = None
        if self.config.cuda_graph_scope and 'mixer' not in self.config.cuda_graph_scope:
            hidden_states, context = self._forward_mixer(*args, **kwargs)
            args = (hidden_states,)
            kwargs = {}

        assert (kwargs.get('inference_context') is None) and (
            kwargs.get('packed_seq_params') is None
        ), (
            "CUDA graph accepts only Tensor inputs. "
            "inference_context and packed_seq_params are excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )

        cuda_graph_output = list(super()._te_cuda_graph_replay(*args, **kwargs))

        if kwargs.get('context') is not None:
            context = cuda_graph_output.pop()

        if (
            not self.config.cuda_graph_scope
            or (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope)
            or (self.is_moe_layer and 'moe' in self.config.cuda_graph_scope)
        ):
            # CUDA Graph captures the whole MLP/MoE part. CUDA Graph output is the layer output.
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
        elif self.is_moe_layer and 'moe_router' in self.config.cuda_graph_scope:
            # CUDA Graph partially captures the MoE.
            # The rest of the layer should go to the normal pass.
            shared_expert_output, routing_map, residual = None, None, None
            mlp_residual = cuda_graph_output.pop()
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                # The shared expert output is the fourth element in the CUDA graph output.
                shared_expert_output = cuda_graph_output.pop()

            # Split cudagraph outputs into function outputs and attribute outputs, and
            # process them separately. Function outputs should have three tensors.
            func_output, attr_outputs = cuda_graph_output[:3], cuda_graph_output[3:]
            if 'moe_preprocess' in self.config.cuda_graph_scope:
                hidden_states, probs, residual = func_output
                valid_cudagraph_attrs = self.mlp.token_dispatcher.valid_cudagraph_attrs
                assert len(attr_outputs) == len(
                    valid_cudagraph_attrs
                ), f"attr_outputs: {len(attr_outputs)} != {len(valid_cudagraph_attrs)}"
                for i, attr_name in enumerate(valid_cudagraph_attrs):
                    hier_attr_name = attr_name.split('.')
                    attr = self.mlp.token_dispatcher
                    for name in hier_attr_name[:-1]:
                        attr = getattr(attr, name)
                    setattr(attr, hier_attr_name[-1], attr_outputs[i])
            else:
                hidden_states, probs, routing_map = func_output
                assert not attr_outputs, "cuda_graph_attr_outputs should be empty"

            # Resume the MoELayer forward pass from the end of the CUDA graph scope.
            # The MoE layer will skip redundant computations when we pass in the calculated values
            # through the keyword arguments. See MoELayer.forward docstring for more details.
            nvtx_range_push(suffix="mlp")
            self.mlp.cudagraph_tensor_store.set(
                hidden_states=hidden_states,
                probs=probs,
                routing_map=routing_map,
                residual=residual,
                shared_expert_output=shared_expert_output,
            )
            mlp_output_with_bias = self.mlp(hidden_states)
            self.mlp.cudagraph_tensor_store.clear()
            nvtx_range_pop(suffix="mlp")

            output = self._forward_post_mlp(mlp_output_with_bias, mlp_residual)
        else:
            # CUDA Graph does not capture the MLP/MoE part at all.
            output = self._forward_mlp(*cuda_graph_output)
        return output, context

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Helper function to get tensor arguments for TE CUDA graph."""
        cudagraph_args, cudagraph_kwargs = super()._get_te_cuda_graph_replay_args(*args, **kwargs)

        assert (
            len(cudagraph_args) == 1
        ), "Exactly one positional argument `hidden_states` is expected."
        hidden_states = cudagraph_args[0]

        try:
            import transformer_engine.pytorch as te  # pylint: disable=unused-import

            def get_zero_attention_mask(slen_per_tpcp, micro_batch_size):
                sequence_parallel = self.config.sequence_parallel
                tensor_model_parallel_size = self.config.tensor_model_parallel_size
                slen_per_cp = (
                    slen_per_tpcp * tensor_model_parallel_size
                    if sequence_parallel
                    else slen_per_tpcp
                )
                slen = slen_per_cp * self.config.context_parallel_size
                return torch.zeros(
                    (micro_batch_size, 1, slen_per_cp, slen),
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                )

            if not is_te_min_version("1.10.0"):
                # TE version < 1.10.0 does not support keyword arguments with CUDA graph.
                for k, v in cudagraph_kwargs.items():
                    if k == "attention_mask":
                        if v is not None:
                            cudagraph_args.append(v)
                            cudagraph_kwargs[k] = None
                        else:
                            cudagraph_args.append(
                                get_zero_attention_mask(
                                    hidden_states.size(0), hidden_states.size(1)
                                )
                            )
                    elif k != 'is_first_microbatch':
                        assert v is None, "Keyword Arguments not supported with CUDA graph."
            elif (
                'attention_mask' in cudagraph_kwargs and cudagraph_kwargs['attention_mask'] is None
            ):
                # The attention_mask can be None when there is no padding to the input sequence.
                # However, an attention_mask Tensor must be passed into cudagraph for replay, so
                # we create an equivalent zero Tensor as the attention_mask.
                cudagraph_kwargs["attention_mask"] = get_zero_attention_mask(
                    hidden_states.size(0), hidden_states.size(1)
                )
        except ImportError:
            raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
        return tuple(cudagraph_args), cudagraph_kwargs

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return True
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                (kwargs.get('inference_context') is not None)
                or (kwargs.get('inference_params') is not None)
            )
            and 'full_iteration' not in self.config.cuda_graph_scope
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                # it can happen that non-decode steps have a token count greater than the max
                # supported cuda graph token count. In that case this flag will be set to
                # False by initialize_attention, and we should not use cuda graphs.
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()
            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            # Inference mode.
            if kwargs.get('inference_context') is not None:
                # dynamic_inference_decode_only is not a real argument to forward, it is only used
                # to differentiate the cuda graph used for decode from the one used for non-decode
                # inference.
                kwargs["dynamic_inference_decode_only"] = kwargs[
                    'inference_context'
                ].is_decode_only()
        return super().__call__(*args, **kwargs)
