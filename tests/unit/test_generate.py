"""Unit tests for the generation loop."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from infer.engine.generate import GenerationResult, GenerationTiming, generate
from infer.engine.sampler import SamplingParams

# ---------------------------------------------------------------------------
# Mock model / tokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Simple tokenizer for testing: maps token IDs to characters."""

    def __init__(self, vocab_size: int, eos_token_ids: set[int]) -> None:
        self._vocab_size = vocab_size
        self._eos_token_ids = eos_token_ids

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos_token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ids = token_ids
        if skip_special_tokens:
            ids = [t for t in ids if t not in self._eos_token_ids]
        return "".join(chr(ord("A") + (t % 26)) for t in ids)


class MockModel(nn.Module):
    """Model that always returns logits favoring a fixed token."""

    def __init__(self, vocab_size: int, fixed_next_token: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._fixed_next_token = fixed_next_token

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self._vocab_size)
        logits[:, :, self._fixed_next_token] = 10.0
        return logits


class SequenceMockModel(nn.Module):
    """Model that emits tokens from a predefined sequence (greedy)."""

    def __init__(self, vocab_size: int, sequence: list[int]) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._sequence = sequence
        self._step = 0

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self._vocab_size)
        idx = min(self._step, len(self._sequence) - 1)
        logits[:, -1, self._sequence[idx]] = 10.0
        self._step += 1
        return logits


class UniformMockModel(nn.Module):
    """Model that returns all-zero logits (uniform distribution)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        return torch.zeros(batch, seq_len, self._vocab_size)


# ---------------------------------------------------------------------------
# GenerationTiming properties
# ---------------------------------------------------------------------------


class TestGenerationTiming:
    def test_decode_time_s(self) -> None:
        t = GenerationTiming(prefill_time_s=0.1, decode_times_s=[0.2, 0.3, 0.4])
        assert t.decode_time_s == pytest.approx(0.9)

    def test_total_time_s(self) -> None:
        t = GenerationTiming(prefill_time_s=0.1, decode_times_s=[0.2, 0.3])
        assert t.total_time_s == pytest.approx(0.6)

    def test_empty_decode(self) -> None:
        t = GenerationTiming(prefill_time_s=0.5, decode_times_s=[])
        assert t.decode_time_s == 0.0
        assert t.total_time_s == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------


class TestBasicGeneration:
    def test_greedy_fixed_token(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        result = generate(model, tokenizer, [0, 1, 2], params, device="cpu")  # type: ignore[arg-type]
        assert result.generated_tokens == 5
        assert result.token_ids == [7, 7, 7, 7, 7]
        assert result.finish_reason == "length"
        assert result.prompt_tokens == 3

    def test_returns_generation_result(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert isinstance(result, GenerationResult)
        assert isinstance(result.timing, GenerationTiming)
        assert result.prompt_tokens == 1
        assert result.generated_tokens == 3
        assert len(result.text) > 0


# ---------------------------------------------------------------------------
# EOS stopping
# ---------------------------------------------------------------------------


class TestEOSStopping:
    def test_stops_on_eos(self) -> None:
        # Emit tokens: 5, 5, 99 (EOS). Should stop after 3 generated tokens.
        model = SequenceMockModel(vocab_size=100, sequence=[5, 5, 99])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "eos"
        assert result.generated_tokens == 3
        assert result.token_ids == [5, 5, 99]

    def test_stops_on_any_eos(self) -> None:
        # Multiple EOS IDs: stops on any match.
        model = SequenceMockModel(vocab_size=100, sequence=[5, 5, 88])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={88, 99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "eos"
        assert result.generated_tokens == 3

    def test_eos_on_first_token(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "eos"
        assert result.generated_tokens == 1


# ---------------------------------------------------------------------------
# Stop string stopping
# ---------------------------------------------------------------------------


class TestStopStringStopping:
    def test_stop_string_triggers(self) -> None:
        # Tokens: 0, 1, 2, 3 -> "ABCD". Stop on "CD".
        # After token 3: text = "ABCD", "CD" found -> stop.
        model = SequenceMockModel(vocab_size=100, sequence=[0, 1, 2, 3, 4])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["CD"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        assert "CD" not in result.text
        assert result.text == "AB"
        assert result.generated_tokens == 4

    def test_stop_string_truncates_text(self) -> None:
        # Tokens: 0, 1, 2 -> "ABC". Stop on "BC".
        # After token 2: text = "ABC", "BC" found -> stop.
        model = SequenceMockModel(vocab_size=100, sequence=[0, 1, 2, 3])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["BC"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        assert result.text == "A"

    def test_empty_stop_list(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=2)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3, stop=[])
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "length"
        assert result.generated_tokens == 3

    def test_none_stop(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=2)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "length"
        assert result.generated_tokens == 3

    def test_stop_string_on_first_token(self) -> None:
        # First generated token is 0 -> decoded as "A". Stop on "A".
        model = MockModel(vocab_size=100, fixed_next_token=0)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["A"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        assert result.generated_tokens == 1
        assert result.text == ""

    def test_multiple_stop_strings_earliest_wins(self) -> None:
        # Tokens: 0, 1, 2, 3 -> "ABCD". stop=["CD", "AB"].
        # "AB" appears at index 0, "CD" at index 2 -> truncate at "AB".
        model = SequenceMockModel(vocab_size=100, sequence=[0, 1, 2, 3, 4])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["CD", "AB"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        # "AB" is the earlier match (index 0), so text is truncated there.
        assert result.text == ""

    def test_multiple_stop_strings_second_in_list_matches_first_in_text(self) -> None:
        # Tokens: 0, 1, 2, 3 -> "ABCD". stop=["CD", "B"].
        # "B" appears at index 1, "CD" at index 2 -> truncate at "B".
        model = SequenceMockModel(vocab_size=100, sequence=[0, 1, 2, 3, 4])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["CD", "B"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        assert result.text == "A"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        # Uniform logits so the seed meaningfully affects token selection.
        model = UniformMockModel(vocab_size=50)
        tokenizer = MockTokenizer(vocab_size=50, eos_token_ids={999})
        params = SamplingParams(temperature=1.0, seed=42, max_new_tokens=10)
        r1 = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        r2 = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert r1.token_ids == r2.token_ids

    def test_different_seeds_differ(self) -> None:
        # With 50 uniform tokens and 10 samples, collision probability is ~10^-17.
        model = UniformMockModel(vocab_size=50)
        tokenizer = MockTokenizer(vocab_size=50, eos_token_ids={999})
        p1 = SamplingParams(temperature=1.0, seed=1, max_new_tokens=10)
        p2 = SamplingParams(temperature=1.0, seed=999, max_new_tokens=10)
        r1 = generate(model, tokenizer, [0], p1, device="cpu")  # type: ignore[arg-type]
        r2 = generate(model, tokenizer, [0], p2, device="cpu")  # type: ignore[arg-type]
        assert r1.token_ids != r2.token_ids

    def test_greedy_deterministic(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        r1 = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        r2 = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert r1.token_ids == r2.token_ids


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TestTiming:
    def test_prefill_positive(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.timing.prefill_time_s > 0

    def test_decode_times_count(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=5)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        # First token is part of prefill, remaining are decode steps.
        assert len(result.timing.decode_times_s) == result.generated_tokens - 1

    def test_total_time_consistent(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.timing.total_time_s >= result.timing.prefill_time_s
        assert result.timing.total_time_s >= result.timing.decode_time_s

    def test_decode_times_with_eos(self) -> None:
        # EOS after 3 tokens: 1 prefill + 2 decode steps.
        model = SequenceMockModel(vocab_size=100, sequence=[5, 5, 99])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.generated_tokens == 3
        assert len(result.timing.decode_times_s) == 2  # 3 - 1

    def test_decode_times_with_stop_string(self) -> None:
        # Tokens: 0, 1, 2 -> "ABC". Stop on "BC" after 3 tokens.
        # 1 prefill + 2 decode steps.
        model = SequenceMockModel(vocab_size=100, sequence=[0, 1, 2, 3])
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["BC"])
        result = generate(model, tokenizer, [50], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "stop"
        assert result.generated_tokens == 3
        assert len(result.timing.decode_times_s) == result.generated_tokens - 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_max_new_tokens_one(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=1)
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.generated_tokens == 1
        assert result.finish_reason == "length"
        assert len(result.timing.decode_times_s) == 0

    def test_prompt_length_one(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        result = generate(model, tokenizer, [42], params, device="cpu")  # type: ignore[arg-type]
        assert result.prompt_tokens == 1
        assert result.generated_tokens == 3

    def test_eos_priority_over_stop_string(self) -> None:
        # EOS check runs before stop string check.
        model = MockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=10, stop=["anything"])
        result = generate(model, tokenizer, [0], params, device="cpu")  # type: ignore[arg-type]
        assert result.finish_reason == "eos"

    def test_long_prompt(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=2)
        prompt = list(range(50))
        result = generate(model, tokenizer, prompt, params, device="cpu")  # type: ignore[arg-type]
        assert result.prompt_tokens == 50
        assert result.generated_tokens == 2

    def test_empty_prompt_raises(self) -> None:
        model = MockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer(vocab_size=100, eos_token_ids={99})
        params = SamplingParams(temperature=0.0, max_new_tokens=3)
        with pytest.raises(ValueError, match="prompt_token_ids must not be empty"):
            generate(model, tokenizer, [], params, device="cpu")  # type: ignore[arg-type]
