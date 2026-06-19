# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SFT tokenizer."""
from typing import Dict, List, Union

from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer


class SFTTokenizer(MegatronLegacyTokenizer):
    """SFT Tokenizer.

    Thin wrapper around a HuggingFace tokenizer that relies on the tokenizer's
    own chat template (with assistant-token masking) to build the loss mask.
    """

    def __init__(
        self,
        tokenizer_path: str,
        prompt_format: str = None,
    ):
        """
        Args:
            tokenizer_path (str): Underlying HuggingFace tokenizer path.
            prompt_format (str): Unused; kept for build_tokenizer compatibility.
                The chat template baked into the HuggingFace tokenizer is used.
        """
        super().__init__(tokenizer_path, prompt_format=prompt_format)
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path,
        )

        self._tokenizer = tokenizer
        self._vocab_size = len(tokenizer)
        self._prompt_format = prompt_format

    def tokenize_conversation(self, sample):
        """Convert a conversation to (tokens, assistant loss mask).

        Args:
            sample (Dict): A data sample with key "messages" (List[Dict]) - a
                sequence of system/user/assistant messages - and an optional
                "chat_template_kwargs" dict forwarded to apply_chat_template.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]

        Returns:
            (input_ids, assistant_masks): the token ids and a mask that is 1 on
            assistant tokens (the tokens to train on) and 0 elsewhere. The mask
            is NOT shifted; the causal shift is applied by the dataset.
        """
        out = self._tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_tensors="np",
            return_dict=True,
            **sample.get("chat_template_kwargs", {}),
        )

        return out["input_ids"][0], out["assistant_masks"][0]

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # Inference path: render the prompt with an assistant generation prefix.
            out = self._tokenizer.apply_chat_template(
                text,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="np",
            )
            return out[0].tolist()

        return self._encode(text)

    def _encode(self, text: str):
        """Tokenize text input, w/o chat template"""
        return self._tokenizer.encode(text)

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        pad_id = self._tokenizer.pad_token_id
        return pad_id if pad_id is not None else self._tokenizer.eos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
