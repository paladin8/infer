"""Tests for the tokenizer wrapper.

Tests run against all three dev models.  Each is marked ``@pytest.mark.slow``
and skips gracefully when the model is not accessible.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from infer.loader.tokenizer import Tokenizer

MODEL_IDS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
]


@dataclass
class TokenizerPair:
    """Our wrapper + the raw HF tokenizer for comparison."""

    ours: Tokenizer
    hf: PreTrainedTokenizerBase
    model_id: str


def _load_pair(model_id: str) -> TokenizerPair:
    tok = Tokenizer(model_id)
    hf = AutoTokenizer.from_pretrained(model_id)
    return TokenizerPair(ours=tok, hf=hf, model_id=model_id)


@pytest.fixture(scope="module", params=MODEL_IDS, ids=["llama", "qwen3", "gemma3"])
def pair(request: pytest.FixtureRequest) -> TokenizerPair:
    """Load a tokenizer pair, skipping if the model is unavailable."""
    model_id: str = request.param
    try:
        return _load_pair(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")
        raise  # unreachable — satisfies mypy return check


@pytest.mark.slow
class TestEncode:
    """Test encoding text to token IDs."""

    def test_encode_matches_hf(self, pair: TokenizerPair) -> None:
        text = "Hello, world!"
        assert pair.ours.encode(text) == pair.hf.encode(text)

    def test_encode_without_special_tokens(self, pair: TokenizerPair) -> None:
        text = "Hello, world!"
        ours = pair.ours.encode(text, add_special_tokens=False)
        expected = pair.hf.encode(text, add_special_tokens=False)
        assert ours == expected

    def test_encode_returns_list_of_ints(self, pair: TokenizerPair) -> None:
        ids = pair.ours.encode("test")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self, pair: TokenizerPair) -> None:
        ids = pair.ours.encode("", add_special_tokens=False)
        assert ids == []


@pytest.mark.slow
class TestDecode:
    """Test decoding token IDs back to text."""

    def test_roundtrip(self, pair: TokenizerPair) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        ids = pair.ours.encode(text, add_special_tokens=False)
        decoded = pair.ours.decode(ids)
        assert decoded == text

    def test_decode_matches_hf(self, pair: TokenizerPair) -> None:
        text = "Hello, world!"
        ids = pair.hf.encode(text, add_special_tokens=False)
        assert pair.ours.decode(ids) == pair.hf.decode(ids, skip_special_tokens=True)

    def test_decode_with_special_tokens(self, pair: TokenizerPair) -> None:
        text = "Hello"
        ids = pair.ours.encode(text, add_special_tokens=True)
        decoded = pair.ours.decode(ids, skip_special_tokens=False)
        expected = pair.hf.decode(ids, skip_special_tokens=False)
        assert decoded == expected


@pytest.mark.slow
class TestProperties:
    """Test tokenizer properties match HF directly."""

    def test_eos_token_id_type(self, pair: TokenizerPair) -> None:
        eos = pair.ours.eos_token_id
        assert isinstance(eos, (int, list))
        if isinstance(eos, list):
            assert all(isinstance(i, int) for i in eos)

    def test_eos_matches_hf(self, pair: TokenizerPair) -> None:
        assert pair.ours.eos_token_id == pair.hf.eos_token_id

    def test_bos_matches_hf(self, pair: TokenizerPair) -> None:
        assert pair.ours.bos_token_id == pair.hf.bos_token_id

    def test_vocab_size_positive(self, pair: TokenizerPair) -> None:
        assert pair.ours.vocab_size > 0

    def test_vocab_size_matches_hf(self, pair: TokenizerPair) -> None:
        assert pair.ours.vocab_size == pair.hf.vocab_size


@pytest.mark.slow
class TestModelSpecificProperties:
    """Verify known properties for each dev model."""

    def test_qwen3_no_bos(self) -> None:
        try:
            tok = Tokenizer("Qwen/Qwen3-1.7B")
        except Exception as exc:
            pytest.skip(f"Qwen3 not available: {exc}")
        assert tok.bos_token_id is None

    def test_qwen3_vocab_size(self) -> None:
        try:
            tok = Tokenizer("Qwen/Qwen3-1.7B")
        except Exception as exc:
            pytest.skip(f"Qwen3 not available: {exc}")
        assert tok.vocab_size == 151643

    def test_gemma3_has_bos(self) -> None:
        try:
            tok = Tokenizer("google/gemma-3-1b-it")
        except Exception as exc:
            pytest.skip(f"Gemma3 not available: {exc}")
        assert isinstance(tok.bos_token_id, int)

    def test_gemma3_vocab_size(self) -> None:
        try:
            tok = Tokenizer("google/gemma-3-1b-it")
        except Exception as exc:
            pytest.skip(f"Gemma3 not available: {exc}")
        assert tok.vocab_size == 262144

    def test_llama_has_bos(self) -> None:
        try:
            tok = Tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        except Exception as exc:
            pytest.skip(f"Llama not available: {exc}")
        assert isinstance(tok.bos_token_id, int)

    def test_llama_vocab_size(self) -> None:
        try:
            tok = Tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        except Exception as exc:
            pytest.skip(f"Llama not available: {exc}")
        assert tok.vocab_size == 128256
