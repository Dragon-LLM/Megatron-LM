# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from typing import Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.linear_attention_module_specs import (
    get_linear_attention_module_spec_for_backend,
)
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.dragon.dragon_block import (
    DragonBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.dragon.dragon_attention import SelfDiffAttention, SelfDiffAttentionSubmodules
from megatron.core.dragon.dragon_attention_v2 import SelfDiffAttentionV2, SelfDiffAttentionV2Submodules
from megatron.core.dragon.dragon_gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from megatron.core.dragon.dragon_mamba3 import Mamba3, Mamba3Submodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.dragon.dragon_layer import (
    DragonLayer,
    DragonLayerSubmodules,
    get_dragon_layer_offset,
)
from megatron.core.activations import squared_relu

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False

from megatron.core.extensions.transformer_engine import TELinear, TELayerNormColumnParallelLinear, TEColumnParallelLinear, TERowParallelLinear,TEDotProductAttention, TEColumnParallelGroupedLinear, TERowParallelGroupedLinear

def get_dragon_block_spec(
    config: DragonConfig,
):
    backend = TESpecProvider()
    attention = ModuleSpec(
        module=SelfDiffAttention,
        params={"attn_mask_type": AttnMaskType.causal if not config.intra_doc_masking else AttnMaskType.padding_causal},
        submodules=SelfDiffAttentionSubmodules(
            linear_in=TELayerNormColumnParallelLinear,
            linear_BkBv=TELinear,
            core_attention=TEDotProductAttention,
            q_layernorm=TENorm,
            k_layernorm=TENorm,
        )
    )
    attention_v2 = ModuleSpec(
        module=SelfDiffAttentionV2,
        params={"attn_mask_type": AttnMaskType.causal if not config.intra_doc_masking else AttnMaskType.padding_causal},
        submodules=SelfDiffAttentionV2Submodules(
            linear_in=TELayerNormColumnParallelLinear,
            linear_BkBv=TELinear,
            core_attention=TEDotProductAttention,
            q_layernorm=TENorm,
            k_layernorm=TENorm,
        )
    )
    gdn = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=TELayerNormColumnParallelLinear,
        )
    )
    mamba3 = ModuleSpec(
        module=Mamba3,
        submodules=Mamba3Submodules(
            in_proj=TELayerNormColumnParallelLinear,
            b_norm=TENorm,
            c_norm=TENorm,
            rope_proj=TELinear,
        ),
    )
    # MLP.
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, # no layernorm. it's done as a standalone.
            linear_fc2=TERowParallelLinear,
            activation_func=backend.activation_func() if config.use_te_activation_func else None,
        ),
    )
    # MoE.
    experts = ModuleSpec(
        module=TEGroupedMLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, # no layernorm. it's done as a standalone.
            linear_fc2=TERowParallelGroupedLinear,
        ),
    )
    shared_experts = ModuleSpec(
        module=SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear, # no layernorm. it's done as a standalone.
            linear_fc2=TERowParallelLinear,
            activation_func=backend.activation_func() if config.use_te_activation_func else None,
        ),
    )
    moe = ModuleSpec(
        module=MoELayer,
        params={},
        submodules=MoESubmodules(
            experts=experts,
            shared_experts=shared_experts,
        )
    )
    layer = ModuleSpec(
        module=DragonLayer,
        params={},
        submodules=DragonLayerSubmodules(
            attention=attention,
            attention_v2=attention_v2,
            gdn=gdn,
            mamba3=mamba3,
            mixer_norm=TENorm,
            mixer_proj=TERowParallelLinear,
            pre_mlp_norm=TENorm,
            mlp=mlp,
            moe=moe,
        ),
    )
    dragon_block_spec = DragonBlockSubmodules(
        layer_specs=[layer] * config.num_layers,
        final_layer_norm=TENorm,
    )

    return dragon_block_spec
