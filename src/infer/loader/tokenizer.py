"""Thin wrapper around HuggingFace AutoTokenizer.

Provides a stable interface so the rest of the codebase doesn't import
``transformers`` directly.
"""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Tokenizer:
    """Wrapper around a HuggingFace tokenizer.

    Args:
        model_path: Local directory or HF Hub repo ID containing the tokenizer files.
    """

    def __init__(self, model_path: str) -> None:
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The string to tokenize.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of integer token IDs.
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to strip special tokens from the output.

        Returns:
            Decoded string.
        """
        result = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        assert isinstance(result, str)  # single list always returns str
        return result

    @property
    def eos_token_id(self) -> int | list[int]:
        """End-of-sequence token ID(s).

        Returns a list when the model defines multiple EOS tokens
        (e.g. Llama 3.2 uses ``[128001, 128008, 128009]``).
        """
        eos = self._tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no eos_token_id")
        return eos

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID, or ``None`` if not defined."""
        return self._tokenizer.bos_token_id

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (number of tokens the model can produce)."""
        return self._tokenizer.vocab_size
