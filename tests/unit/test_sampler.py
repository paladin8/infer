"""Unit tests for the sampling pipeline."""

from __future__ import annotations

import pytest
import torch

from infer.engine.sampler import (
    SamplingParams,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token,
)

# ---------------------------------------------------------------------------
# SamplingParams validation
# ---------------------------------------------------------------------------


class TestSamplingParamsValidation:
    def test_defaults_valid(self) -> None:
        p = SamplingParams()
        assert p.temperature == 1.0
        assert p.max_new_tokens == 128

    def test_negative_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-1.0)

    def test_top_p_zero(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_top_p_above_one(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)

    def test_top_k_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=0)

    def test_repetition_penalty_zero(self) -> None:
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.0)

    def test_max_new_tokens_zero(self) -> None:
        with pytest.raises(ValueError, match="max_new_tokens"):
            SamplingParams(max_new_tokens=0)

    def test_valid_non_defaults(self) -> None:
        p = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            max_new_tokens=256,
            stop=["###"],
            seed=42,
        )
        assert p.temperature == 0.0
        assert p.seed == 42


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


class TestTemperature:
    def test_identity_at_one(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = apply_temperature(logits, 1.0)
        assert torch.equal(result, logits)

    def test_halves_at_two(self) -> None:
        logits = torch.tensor([2.0, 4.0, -6.0])
        result = apply_temperature(logits, 2.0)
        expected = torch.tensor([1.0, 2.0, -3.0])
        assert torch.allclose(result, expected)

    def test_doubles_at_half(self) -> None:
        logits = torch.tensor([1.0, -2.0, 3.0])
        result = apply_temperature(logits, 0.5)
        expected = torch.tensor([2.0, -4.0, 6.0])
        assert torch.allclose(result, expected)

    def test_preserves_shape_and_dtype(self) -> None:
        logits = torch.randn(1000, dtype=torch.float32)
        result = apply_temperature(logits, 0.8)
        assert result.shape == logits.shape
        assert result.dtype == logits.dtype


# ---------------------------------------------------------------------------
# Top-k filtering
# ---------------------------------------------------------------------------


class TestTopK:
    def test_keeps_k_values(self) -> None:
        logits = torch.arange(10, dtype=torch.float)  # 0..9
        result = apply_top_k(logits, 3)
        finite = result[result != float("-inf")]
        assert finite.numel() == 3

    def test_keeps_largest(self) -> None:
        logits = torch.tensor([1.0, 5.0, 3.0, 9.0, 7.0])
        result = apply_top_k(logits, 3)
        # Largest 3: 9, 7, 5
        assert result[3] == 9.0  # index of 9
        assert result[4] == 7.0  # index of 7
        assert result[1] == 5.0  # index of 5
        assert result[0] == float("-inf")
        assert result[2] == float("-inf")

    def test_k_ge_vocab_noop(self) -> None:
        logits = torch.randn(10)
        result = apply_top_k(logits, 10)
        assert torch.equal(result, logits)
        result2 = apply_top_k(logits, 100)
        assert torch.equal(result2, logits)

    def test_k_one_keeps_max(self) -> None:
        logits = torch.tensor([1.0, 5.0, 3.0])
        result = apply_top_k(logits, 1)
        finite = result[result != float("-inf")]
        assert finite.numel() == 1
        assert finite.item() == 5.0

    def test_duplicate_boundary_keeps_exactly_k(self) -> None:
        # Three tokens share the value at the k boundary.
        logits = torch.tensor([1.0, 3.0, 3.0, 3.0, 2.0])
        result = apply_top_k(logits, 2)
        finite = result[result != float("-inf")]
        assert finite.numel() == 2

    def test_all_identical_keeps_exactly_k(self) -> None:
        logits = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        result = apply_top_k(logits, 1)
        finite = result[result != float("-inf")]
        assert finite.numel() == 1


# ---------------------------------------------------------------------------
# Top-p (nucleus) filtering
# ---------------------------------------------------------------------------


class TestTopP:
    def test_noop_at_one(self) -> None:
        logits = torch.randn(100)
        result = apply_top_p(logits, 1.0)
        assert torch.equal(result, logits)

    def test_keeps_nucleus(self) -> None:
        # Token 0 has ~90% of the probability mass after softmax.
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
        result = apply_top_p(logits, 0.9)
        # Token 0 alone should exceed 0.9 cumulative probability.
        finite = result[result != float("-inf")]
        assert finite.numel() >= 1
        # The top token must be kept.
        assert result[0] != float("-inf")

    def test_small_p_keeps_few(self) -> None:
        logits = torch.tensor([10.0, 1.0, 0.5, 0.1, -1.0])
        result_small = apply_top_p(logits, 0.01)
        result_large = apply_top_p(logits, 0.99)
        finite_small = (result_small != float("-inf")).sum().item()
        finite_large = (result_large != float("-inf")).sum().item()
        assert finite_small <= finite_large

    def test_exact_nucleus_count(self) -> None:
        # Softmax([2, 1, 0, -1]) ≈ [0.64, 0.24, 0.09, 0.03].
        # Cumulative: [0.64, 0.88, 0.97, 1.00].
        # p=0.9: first 3 tokens cover 0.97 >= 0.9 -> keep 3.
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        result = apply_top_p(logits, 0.9)
        finite = result[result != float("-inf")]
        assert finite.numel() == 3

    def test_exact_nucleus_count_tight(self) -> None:
        # p=0.6: token 0 alone covers 0.64 >= 0.6 -> keep 1.
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        result = apply_top_p(logits, 0.6)
        finite = result[result != float("-inf")]
        assert finite.numel() == 1
        assert result[0] != float("-inf")  # the top token is kept

    def test_handles_neg_inf_entries(self) -> None:
        logits = torch.tensor([5.0, float("-inf"), 3.0, float("-inf"), 1.0])
        result = apply_top_p(logits, 0.9)
        # Should not crash; -inf entries stay -inf.
        assert result[1] == float("-inf")
        assert result[3] == float("-inf")


# ---------------------------------------------------------------------------
# Repetition penalty
# ---------------------------------------------------------------------------


class TestRepetitionPenalty:
    def test_noop_at_one(self) -> None:
        logits = torch.tensor([1.0, -2.0, 3.0])
        result = apply_repetition_penalty(logits, [0, 1, 2], 1.0)
        assert torch.equal(result, logits)

    def test_positive_logit_divided(self) -> None:
        logits = torch.tensor([4.0, 2.0, 1.0])
        result = apply_repetition_penalty(logits, [0], 2.0)
        assert result[0].item() == pytest.approx(2.0)  # 4 / 2
        assert result[1].item() == pytest.approx(2.0)  # unchanged
        assert result[2].item() == pytest.approx(1.0)  # unchanged

    def test_negative_logit_multiplied(self) -> None:
        logits = torch.tensor([-4.0, 2.0, 1.0])
        result = apply_repetition_penalty(logits, [0], 2.0)
        assert result[0].item() == pytest.approx(-8.0)  # -4 * 2
        assert result[1].item() == pytest.approx(2.0)

    def test_tokens_not_in_context_unchanged(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = apply_repetition_penalty(logits, [0, 2], 2.0)
        assert result[1].item() == pytest.approx(2.0)
        assert result[3].item() == pytest.approx(4.0)

    def test_duplicates_no_double_penalty(self) -> None:
        logits = torch.tensor([4.0, 2.0])
        result = apply_repetition_penalty(logits, [0, 0, 0], 2.0)
        assert result[0].item() == pytest.approx(2.0)  # divided once, not thrice

    def test_empty_context(self) -> None:
        logits = torch.tensor([1.0, 2.0, 3.0])
        result = apply_repetition_penalty(logits, [], 2.0)
        assert torch.equal(result, logits)


# ---------------------------------------------------------------------------
# sample_token integration
# ---------------------------------------------------------------------------


class TestSampleToken:
    def test_greedy_returns_argmax(self) -> None:
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits, [], params)
        assert token == 1  # index of 5.0

    def test_greedy_with_repetition_penalty(self) -> None:
        logits = torch.tensor([1.0, 5.0, 4.9])
        # Penalize token 1 (the argmax) heavily.
        params = SamplingParams(temperature=0.0, repetition_penalty=100.0)
        token = sample_token(logits, [1], params)
        assert token == 2  # token 1 is penalized, token 2 (4.9) wins

    def test_deterministic_with_seed(self) -> None:
        logits = torch.randn(1000)
        params = SamplingParams(temperature=0.8, seed=42)
        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(42)
        t1 = sample_token(logits, [], params, gen1)
        t2 = sample_token(logits, [], params, gen2)
        assert t1 == t2

    def test_different_seeds_differ(self) -> None:
        # Use a uniform-ish distribution so different seeds are likely to produce different tokens.
        logits = torch.zeros(1000)
        params_a = SamplingParams(temperature=1.0)
        params_b = SamplingParams(temperature=1.0)
        gen_a = torch.Generator(device="cpu")
        gen_a.manual_seed(1)
        gen_b = torch.Generator(device="cpu")
        gen_b.manual_seed(999)
        t_a = sample_token(logits, [], params_a, gen_a)
        t_b = sample_token(logits, [], params_b, gen_b)
        # With 1000 uniform options, probability of collision is 0.1%.
        assert t_a != t_b

    def test_full_pipeline_returns_valid_token(self) -> None:
        logits = torch.randn(32000)
        params = SamplingParams(
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=123,
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        token = sample_token(logits, list(range(100)), params, gen)
        assert 0 <= token < 32000

    def test_uniform_sampling_approximately_uniform(self) -> None:
        vocab_size = 10
        logits = torch.zeros(vocab_size)
        params = SamplingParams(temperature=1.0)
        counts = torch.zeros(vocab_size, dtype=torch.long)
        n_samples = 10000
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        for _ in range(n_samples):
            t = sample_token(logits, [], params, gen)
            counts[t] += 1
        # Each token should appear ~1000 times.  Allow generous tolerance.
        expected = n_samples / vocab_size
        for c in counts:
            assert c.item() > expected * 0.5, (
                f"Token appeared {c.item()} times, expected ~{expected}"
            )
            assert c.item() < expected * 1.5, (
                f"Token appeared {c.item()} times, expected ~{expected}"
            )
