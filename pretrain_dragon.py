# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.dragon import DragonModel
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from gpt_builders import dragon_builder

import transformers

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
g_scheduler = {'scheduler': None}

import traceback, pickle
import torch
old_wrap = torch.distributed.checkpoint.utils._wrap_exception                                                                                                                        
def patched_wrap(exc):
    traceback.print_exception(exc)  # Print the REAL error                                                                                                                           
    exc.__traceback__ = None        # Strip traceback so pickle works                                                                                                                
    return old_wrap(exc)                                                                                                                                                             
torch.distributed.checkpoint.utils._wrap_exception = patched_wrap                                                                                                                    


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args, config_class=DragonConfig)
    # TODO: this is pretty hacky, find a better way
    if (not is_first_or_last_pipeline_stage(vp_stage) 
        and not (args.create_cu_seqlens_in_dataloader and args.pipeline_model_parallel_size > 1 and args.num_virtual_stages_per_pipeline_rank > 1)
        and 
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        )

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[DragonModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (DragonModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def debug_batch(tokens, position_ids, cu_seqlens, max_seqlen):
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    # debug to check tokens
    seq_tokens = tokens[0]
    seq_pos_ids = position_ids[0]

    # 2. Find indices where position_id resets to 0
    # .nonzero() gives us the indices where the condition is True
    reset_indices = (seq_pos_ids == 0).nonzero(as_tuple=False).flatten().tolist()

    print_rank_0(f"Found document starts at indices: {reset_indices}")

    # 3. Iterate through these resets and print context
    context_window = 10  # How many tokens to show before/after

    for idx in reset_indices:
        # Define the slice bounds
        start = max(0, idx - context_window)
        end = min(len(seq_tokens), idx + context_window)
        
        # Slice the tensors
        # 'pre_tokens' are the end of the previous document
        # 'post_tokens' are the start of the new document (starting at idx)
        pre_tokens = seq_tokens[start:idx]
        post_tokens = seq_tokens[idx:end]
        
        # Decode to string
        # We use replace('\n', '\\n') so you can see newlines in the logs clearly
        pre_text = tok.decode(pre_tokens).replace('\n', '\\n')
        post_text = tok.decode(post_tokens).replace('\n', '\\n')
        
        print_rank_0(f"--- Reset at Index {idx} ---")
        print_rank_0(f"   End of Doc A: \"...{pre_text}\"")
        print_rank_0(f"   Start of Doc B: \"{post_text}...\"")
        print_rank_0("-" * 30)
    
    # 1. Prepare Data
    # cu_seqlens describes the entire flattened batch. 
    # We flatten position_ids to match the dimension of cu_seqlens.
    flat_pos_ids = position_ids.view(-1) 
    cu_seqlens_cpu = cu_seqlens.cpu().view(-1)

    # 2. Derive expected boundaries from Position IDs (The "Ground Truth")
    # We look for where pos_id is 0. 
    # Note: This finds the START of every document.
    pos_reset_indices = (flat_pos_ids == 0).nonzero(as_tuple=False).flatten().cpu()

    # 3. Derive boundaries from cu_seqlens (The "Metadata")
    # cu_seqlens is [0, len1, len1+len2, ...]. 
    # So the starts are everything except the very last element.
    cu_start_indices = cu_seqlens_cpu[:-1]

    # 4. Compare specific Start Indices
    print_rank_0(f"\n--- Checking cu_seqlens Consistency ---")
    print_rank_0(f"Num documents (pos_ids): {len(pos_reset_indices)}")
    print_rank_0(f"Num documents (cu_seq):  {len(cu_start_indices)}")

    # Check if they match exactly
    if torch.equal(pos_reset_indices, cu_start_indices):
        print_rank_0("✅ cu_seqlens boundaries match position_ids exactly.")
    else:
        print_rank_0("❌ MISMATCH DETECTED!")
        print_rank_0(f"   Pos ID Resets (first 10): {pos_reset_indices[:10].tolist()}")
        print_rank_0(f"   Cu Seq Starts (first 10): {cu_start_indices[:10].tolist()}")
        
        # Identify the first mismatch index
        min_len = min(len(pos_reset_indices), len(cu_start_indices))
        diff = (pos_reset_indices[:min_len] != cu_start_indices[:min_len]).nonzero(as_tuple=False)
        if diff.numel() > 0:
            first_diff = diff[0].item()
            print_rank_0(f"   First divergence at doc index {first_diff}:")
            print_rank_0(f"   -> Pos ID says start is at: {pos_reset_indices[first_diff].item()}")
            print_rank_0(f"   -> cu_seqlens says start is at: {cu_start_indices[first_diff].item()}")

    # 5. Check Max Seqlen
    # Calculate individual lengths: cu_seqlens[i+1] - cu_seqlens[i]
    doc_lengths = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    calculated_max = doc_lengths.max().item()
    provided_max = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()

    print_rank_0(f"\n--- Checking max_seqlen Consistency ---")
    print_rank_0(f"Calculated Max (from cu_seqlens): {calculated_max}")
    print_rank_0(f"Provided Max (variable):          {provided_max}")

    if calculated_max == provided_max:
        print_rank_0("✅ max_seqlen is correct.")
    else:
        print_rank_0(f"❌ max_seqlen is WRONG. Diff: {provided_max - calculated_max}")

    print_rank_0("-" * 30)


def forward_step(data_iterator, model: DragonModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (DragonModel): The Dragon Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen = get_batch(data_iterator, vp_stage)
        #print(f"PP rank : {parallel_state.get_pipeline_model_parallel_rank()} VPP rank : {parallel_state.get_virtual_pipeline_model_parallel_rank()} got batch with tokens shape {tokens.shape if tokens is not None else 'None'} and position_ids shape {position_ids.shape if position_ids is not None else 'None'} and cu_seqlens shape {cu_seqlens.shape if cu_seqlens is not None else 'None'} and max_seqlen {max_seqlen if max_seqlen is not None else 'None'}")
    timers('batch-generator').stop()

    #debug_batch(tokens, position_ids, cu_seqlens, max_seqlen)

    # check init mean & std
    """with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None or p.numel() == 0:
                continue
            t = p.detach().float()
            mean = t.mean().item()
            std  = t.std(unbiased=False).item()
            print_rank_0(f"{name:60s} shape={tuple(p.shape)} mean={mean:+.4e} std={std:.4e}")"""

    packed_seq_params = None
    if cu_seqlens is not None:
        assert model.config.intra_doc_masking
        cu_seqlens = cu_seqlens.squeeze(0).to(torch.int32)
        max_seqlen = max_seqlen.item()
        packed_seq_params = PackedSeqParams(qkv_format='thd', position_ids=position_ids.squeeze(0), cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen)

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels, window_size=(g_scheduler["scheduler"].get_wsize(), 0), packed_seq_params=packed_seq_params)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, window_size=(g_scheduler["scheduler"].get_wsize(), 0), packed_seq_params=packed_seq_params,
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, window_size=(g_scheduler["scheduler"].get_wsize(), 0), packed_seq_params=packed_seq_params,
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args, config_class=DragonConfig)
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        or (args.create_cu_seqlens_in_dataloader and args.pipeline_model_parallel_size > 1 and args.num_virtual_stages_per_pipeline_rank > 1)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    if args.create_cu_seqlens_in_dataloader:
        assert not args.eod_mask_loss, "we don't want to mask the loss of the first sequence!"

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        create_cu_seqlens=args.create_cu_seqlens_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, dragon_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
        g_scheduler=g_scheduler,
    )
