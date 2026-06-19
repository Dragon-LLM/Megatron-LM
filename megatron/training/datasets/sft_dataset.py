# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: SFTLowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: SFTLowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> SFTLowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _getitem_packed(self, record, max_seq_len, L, tokenizer) -> Dict[str, Any]:
        """Build a training sample from a pre-tokenized packed record.

        record = {'input_ids', 'loss_mask', 'cu_seqlens', 'tokens_length'} where
        cu_seqlens are document boundaries WITHIN the pack (in token units of the
        unshifted sequence). We pad to L = max_seq_len + 1, causal-shift, build
        per-document position_ids (reset at each boundary), and pass cu_seqlens +
        max_seqlen through so get_batch can populate packed_seq_params and the
        Mamba/attention kernels reset state per document.
        """
        ids  = list(record['input_ids'])
        mask = list(record['loss_mask'])
        cu   = list(record['cu_seqlens'])
        num_tokens = int(record['tokens_length'])

        # truncate to L (packer already bounds to <= max_seq_len, but be safe)
        if len(ids) > L:
            ids = ids[:L]; mask = mask[:L]
            cu = [c for c in cu if c < L] + [L]
        # pad to L
        pad = L - len(ids)
        if pad > 0:
            ids  = ids  + [tokenizer.pad] * pad
            mask = mask + [0.0] * pad

        ids  = torch.as_tensor(ids,  dtype=torch.long)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        # causal shift: ids has length L = max_seq_len + 1; tokens/labels/loss_mask
        # are length max_seq_len. cu_seqlens (built on the unshifted ids in [0, L])
        # must therefore be expressed in the SHIFTED model-input coordinate
        # [0, max_seq_len] to match the pretraining contract (where cu_seqlens is
        # computed on the already-shifted `tokens`). We map an unshifted boundary
        # b (start of a document in `ids`) to the shifted index max(0, b-?)... in
        # practice the shift drops index 0, so boundary b in ids maps to b in the
        # labels frame; we clamp the final boundary to max_seq_len and dedup.
        tokens    = ids[:-1].contiguous()
        labels    = ids[1:].contiguous()
        loss_mask = mask[1:].contiguous()

        # Boundaries in the shifted (model-input) coordinate: clamp to max_seq_len
        # and drop any degenerate (zero-length) segments after clamping.
        cu_shift = []
        for c in cu:
            c = min(int(c), max_seq_len)
            if not cu_shift or c != cu_shift[-1]:
                cu_shift.append(c)
        if cu_shift[0] != 0:
            cu_shift = [0] + cu_shift
        if cu_shift[-1] != max_seq_len:
            cu_shift.append(max_seq_len)

        # per-document position_ids: 0..len-1 within each packed doc (shifted frame)
        position_ids = torch.zeros(max_seq_len, dtype=torch.long)
        for a, b in zip(cu_shift[:-1], cu_shift[1:]):
            if b > a:
                position_ids[a:b] = torch.arange(b - a, dtype=torch.long)

        cu_seqlens = torch.as_tensor(cu_shift, dtype=torch.int32)
        seg = (cu_seqlens[1:] - cu_seqlens[:-1])
        max_seqlen = torch.tensor(int(seg.max().item()) if seg.numel() else max_seq_len,
                                  dtype=torch.int32)

        ret = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'num_tokens': torch.tensor(num_tokens),
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
        if self.config.create_attention_mask:
            ret['attention_mask'] = None
        return ret

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length
        L = max_seq_len + 1

        record = self.dataset[int(self.indices[idx % len(self.indices)])]

        # --- Packed (pre-tokenized) path -------------------------------------
        # Offline packer (pack_sft.py) writes records with 'input_ids',
        # 'loss_mask', and 'cu_seqlens' (document boundaries within the pack).
        # These are already tokenized; we only shift + pad here and surface
        # cu_seqlens so the Mamba kernel resets state at each doc boundary.
        if 'input_ids' in record:
            return self._getitem_packed(record, max_seq_len, L, tokenizer)

        # --- Legacy on-the-fly tokenization path (single conversation) -------
        conversation_list = record
        num_tokens = conversation_list['tokens_length']
        tokens_conv, loss_mask = tokenizer.tokenize_conversation(conversation_list)
        # truncation
        if len(tokens_conv) > L:
            tokens_conv = tokens_conv[:L]
            loss_mask = loss_mask[:L]
        # padding
        padding_len = L - len(tokens_conv)
        if padding_len:
            tokens_conv = np.array(list(tokens_conv) + [tokenizer.pad] * padding_len, dtype=np.int64)
            loss_mask   = np.array(list(loss_mask)   + [0.]            * padding_len, dtype=np.float32)
        else:
            tokens_conv = np.array(tokens_conv, dtype=np.int64)
            loss_mask   = np.array(loss_mask, dtype=np.float32)
        # torch conversion
        tokens_conv = torch.as_tensor(tokens_conv, dtype=torch.long)
        loss_mask   = torch.as_tensor(loss_mask,   dtype=torch.float32)
        # shift for causal lm
        tokens       = tokens_conv[:-1].contiguous()
        labels       = tokens_conv[1:].contiguous()
        loss_mask    = loss_mask[1:].contiguous()
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': labels,
                'attention_mask': None,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
                'num_tokens': torch.tensor(num_tokens),
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
                'num_tokens': torch.tensor(num_tokens),
            }

        return ret
