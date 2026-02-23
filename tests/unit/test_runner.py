"""Unit tests for the ModelRunner."""

from __future__ import annotations

import dataclasses
import time

import torch
from torch import Tensor, nn

from infer.cache.simple import KVCache
from infer.engine.config import EngineConfig
from infer.engine.request import Request, RequestState, StepOutput
from infer.engine.runner import ModelRunner
from infer.engine.sampler import SamplingParams
from infer.loader.config import ModelConfig

# ---------------------------------------------------------------------------
# Mock model / tokenizer
# ---------------------------------------------------------------------------

_MOCK_CONFIG = ModelConfig(
    model_type="llama",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
    max_position_embeddings=256,
    head_dim=8,
)


class MockTokenizer:
    """Simple tokenizer: maps token IDs to characters, A-Z cycling."""

    def __init__(self, eos_token_ids: set[int] | None = None) -> None:
        self._eos_token_ids = eos_token_ids or {99}

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos_token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        ids = token_ids
        if skip_special_tokens:
            ids = [t for t in ids if t not in self._eos_token_ids]
        return "".join(chr(ord("A") + (t % 26)) for t in ids)


class BatchedMockModel(nn.Module):
    """Mock model that supports padding_mask and KV cache.

    Always returns logits favoring ``fixed_next_token`` at every position.
    """

    def __init__(self, vocab_size: int, fixed_next_token: int) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._fixed_next_token = fixed_next_token
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        logits[:, :, self._fixed_next_token] = 10.0
        return logits


class BatchedSequenceMockModel(nn.Module):
    """Mock model that emits tokens from a per-batch-slot sequence.

    ``sequences`` is a list of token-ID lists, one per batch slot.
    Each slot has its own step counter.
    """

    def __init__(self, vocab_size: int, sequences: list[list[int]]) -> None:
        super().__init__()
        self.config = dataclasses.replace(_MOCK_CONFIG, vocab_size=vocab_size)
        self._sequences = sequences
        self._steps: list[int] = [0] * len(sequences)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: Tensor,
        kv_cache: KVCache | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        if kv_cache is not None:
            kv_cache.advance(seq_len)
            out_len = 1 if padding_mask is None else seq_len
        else:
            out_len = seq_len
        logits = torch.zeros(batch, out_len, self.config.vocab_size)
        for i in range(batch):
            if i < len(self._sequences):
                idx = min(self._steps[i], len(self._sequences[i]) - 1)
                logits[i, :, self._sequences[i][idx]] = 10.0
                self._steps[i] += 1
        return logits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_config(**overrides: object) -> EngineConfig:
    defaults: dict[str, object] = {
        "model": "test-model",
        "device": "cpu",
        "dtype": "float16",
        "max_seq_len": 128,
        "max_batch_size": 8,
    }
    defaults.update(overrides)
    return EngineConfig(**defaults)  # type: ignore[arg-type]


def _make_request(
    request_id: str,
    prompt: list[int],
    max_new_tokens: int = 10,
    **kwargs: object,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens, **kwargs),  # type: ignore[arg-type]
        arrival_time_s=time.perf_counter(),
    )


# ---------------------------------------------------------------------------
# Single-request prefill
# ---------------------------------------------------------------------------


class TestSingleRequestPrefill:
    def test_basic_prefill(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.prefill([req])

        assert len(outputs) == 1
        assert outputs[0].request_id == "r1"
        assert outputs[0].token_id == 7
        assert outputs[0].finished is False
        assert req.state is RequestState.DECODE
        assert req.generated_token_ids == [7]
        assert runner._kv_cache is not None
        assert runner._max_prompt_len == 3

    def test_prefill_allocates_kv_cache(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        runner.prefill([req])

        assert runner._kv_cache is not None
        # batch_size=1
        assert runner._kv_cache.k.shape[1] == 1

    def test_prefill_eos_on_first_token(self) -> None:
        """If the first generated token is EOS, request finishes immediately."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        outputs = runner.prefill([req])

        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "eos"
        assert req.state is RequestState.FINISHED


# ---------------------------------------------------------------------------
# Multi-request batched prefill
# ---------------------------------------------------------------------------


class TestBatchedPrefill:
    def test_mixed_prompt_lengths(self) -> None:
        """Batched prefill with prompts of different lengths."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3])
        r2 = _make_request("r2", [10, 20, 30, 40, 50, 60, 70])
        r3 = _make_request("r3", [5, 6, 7, 8, 9])
        outputs = runner.prefill([r1, r2, r3])

        assert len(outputs) == 3
        assert runner._max_prompt_len == 7
        assert runner._prompt_lens == [3, 7, 5]

        # All requests should have 1 generated token and be in DECODE state.
        for req in [r1, r2, r3]:
            assert req.state is RequestState.DECODE
            assert len(req.generated_token_ids) == 1

        # KV cache should have batch_size=3.
        assert runner._kv_cache is not None
        assert runner._kv_cache.k.shape[1] == 3

    def test_all_same_length_batch(self) -> None:
        """All prompts have the same length -- no actual padding needed."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        r2 = _make_request("r2", [4, 5, 6], max_new_tokens=3)
        outputs = runner.prefill([r1, r2])

        assert len(outputs) == 2
        assert runner._max_prompt_len == 3
        mask = runner._padding_mask
        assert mask is not None
        # All positions [0,1,2] should be True for both requests (no padding).
        assert mask[0, :3].all()
        assert mask[1, :3].all()
        # Decode region [3,4,5] starts empty except position 3 (first generated token).
        assert mask[0, 3].item() is True
        assert mask[1, 3].item() is True

    def test_padding_mask_pattern(self) -> None:
        """Verify padding mask has correct True/False pattern."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        r2 = _make_request("r2", [10, 20, 30, 40, 50], max_new_tokens=5)
        runner.prefill([r1, r2])

        mask = runner._padding_mask
        assert mask is not None
        # max_prompt_len=5, max_decode=5, max_total=10
        assert mask.shape == (2, 10)

        # r1: 3 real tokens, positions [0,1,2]=True, [3,4]=False (padding)
        # After prefill, position 5 (max_prompt_len) is also True (first generated token)
        assert mask[0, :3].all()
        assert not mask[0, 3:5].any()  # padding positions
        assert mask[0, 5].item() is True  # first generated token

        # r2: 5 real tokens, all of [0..4]=True
        assert mask[1, :5].all()
        assert mask[1, 5].item() is True  # first generated token

    def test_padding_mask_updates_during_decode(self) -> None:
        """Padding mask marks new token positions as True during decode."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        r2 = _make_request("r2", [10, 20, 30, 40, 50], max_new_tokens=5)
        runner.prefill([r1, r2])

        mask = runner._padding_mask
        assert mask is not None
        # After prefill: position max_prompt_len (5) is True for both.
        assert mask[0, 5].item() is True
        assert mask[1, 5].item() is True
        # Position 6 not yet marked.
        assert mask[0, 6].item() is False
        assert mask[1, 6].item() is False

        runner.decode_step([r1, r2])
        # After first decode: position 6 should be True.
        assert mask[0, 6].item() is True
        assert mask[1, 6].item() is True

        runner.decode_step([r1, r2])
        # After second decode: position 7 should be True.
        assert mask[0, 7].item() is True
        assert mask[1, 7].item() is True


# ---------------------------------------------------------------------------
# Decode step
# ---------------------------------------------------------------------------


class TestDecodeStep:
    def test_basic_decode(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        runner.prefill([req])
        assert len(req.generated_token_ids) == 1

        outputs = runner.decode_step([req])
        assert len(outputs) == 1
        assert outputs[0].token_id == 7
        assert len(req.generated_token_ids) == 2

    def test_decode_multiple_steps(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        runner.prefill([req])

        for _step in range(4):
            outputs = runner.decode_step([req])
            assert outputs[0].token_id == 7

        # prefill (1) + 4 decode = 5 tokens total
        assert len(req.generated_token_ids) == 5

    def test_text_delta_incremental(self) -> None:
        """Each StepOutput has incremental text_delta."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        outputs = runner.prefill([req])

        # First token: text_delta should be the decoded text of token 7
        assert len(outputs[0].text_delta) > 0

        out2 = runner.decode_step([req])
        assert len(out2[0].text_delta) > 0


# ---------------------------------------------------------------------------
# EOS handling
# ---------------------------------------------------------------------------


class TestEOSHandling:
    def test_eos_finishes_request(self) -> None:
        """EOS token during decode finishes the request."""
        # Sequence: first token=5, second token=99 (EOS)
        model = BatchedSequenceMockModel(vocab_size=100, sequences=[[5, 99]])
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=10)
        runner.prefill([req])
        assert req.state is RequestState.DECODE

        outputs = runner.decode_step([req])
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "eos"
        assert req.state is RequestState.FINISHED

    def test_eos_in_batch_other_continues(self) -> None:
        """One request hits EOS, the other continues."""
        model = BatchedSequenceMockModel(
            vocab_size=100,
            sequences=[
                [5, 99],  # r1: hits EOS on step 2
                [5, 5, 5, 5, 5],  # r2: keeps going
            ],
        )
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=10)
        r2 = _make_request("r2", [4, 5], max_new_tokens=10)
        runner.prefill([r1, r2])

        # First decode: r1 hits EOS
        outputs = runner.decode_step([r1, r2])
        assert outputs[0].finished is True  # r1 done
        assert outputs[0].finish_reason == "eos"
        assert outputs[1].finished is False  # r2 still going

        # Second decode: r1 gets noop, r2 continues
        outputs = runner.decode_step([r1, r2])
        assert outputs[0].token_id is None  # r1 noop
        assert outputs[1].finished is False  # r2 still going


# ---------------------------------------------------------------------------
# Max tokens
# ---------------------------------------------------------------------------


class TestMaxTokens:
    def test_max_new_tokens_finishes(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=3)
        runner.prefill([req])  # 1 token generated
        runner.decode_step([req])  # 2 tokens total

        outputs = runner.decode_step([req])  # 3 tokens total → finished
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "length"
        assert req.state is RequestState.FINISHED
        assert len(req.generated_token_ids) == 3

    def test_max_new_tokens_one(self) -> None:
        """max_new_tokens=1: finishes after prefill."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=1)
        outputs = runner.prefill([req])

        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "length"
        assert req.state is RequestState.FINISHED


# ---------------------------------------------------------------------------
# Batch runs until all finish
# ---------------------------------------------------------------------------


class TestBatchRunsUntilAllFinish:
    def test_mixed_finish_times(self) -> None:
        """One request hits EOS early, the other runs to max_new_tokens."""
        model = BatchedSequenceMockModel(
            vocab_size=100,
            sequences=[
                [5, 99],  # r1: hits EOS on step 2
                [5, 5, 5, 5, 5],  # r2: runs to max_new_tokens
            ],
        )
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        r1 = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        r2 = _make_request("r2", [4, 5], max_new_tokens=5)
        runner.prefill([r1, r2])

        # Run 4 decode steps
        all_outputs: list[list[StepOutput]] = []
        for _ in range(4):
            outputs = runner.decode_step([r1, r2])
            all_outputs.append(outputs)

        # r1 should have finished after 1st decode step (EOS)
        assert r1.state is RequestState.FINISHED
        assert r1.finish_reason == "eos"
        assert len(r1.generated_token_ids) == 2  # prefill + 1 decode

        # r2 should have finished after 4th decode step (5 total tokens = max_new_tokens)
        assert r2.state is RequestState.FINISHED
        assert r2.finish_reason == "length"
        assert len(r2.generated_token_ids) == 5


# ---------------------------------------------------------------------------
# Stop string handling
# ---------------------------------------------------------------------------


class TestStopString:
    def test_stop_string_finishes(self) -> None:
        """Stop string detected in generated text finishes the request."""
        # Token 2 → 'C', Token 3 → 'D'. Sequence: [2, 3, ...]
        # After tokens [2, 3], decoded = "CD". Stop on "CD".
        model = BatchedSequenceMockModel(vocab_size=100, sequences=[[2, 3, 4, 5]])
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=10, stop=["CD"])
        runner.prefill([req])
        # After prefill: generated=[2], decoded="C", no stop yet

        outputs = runner.decode_step([req])
        # After decode: generated=[2, 3], decoded="CD", stop detected
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "stop"
        assert req.state is RequestState.FINISHED

    def test_stop_string_during_prefill(self) -> None:
        """Stop string matching the first generated token finishes at prefill."""
        # Token 0 → 'A'. Stop on "A" → finishes immediately after prefill.
        model = BatchedMockModel(vocab_size=100, fixed_next_token=0)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [50, 51], max_new_tokens=10, stop=["A"])
        outputs = runner.prefill([req])

        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "stop"
        assert req.state is RequestState.FINISHED
        # text_delta should be truncated at the stop string (empty, since "A" is the stop).
        assert outputs[0].text_delta == ""

    def test_stop_string_text_delta_truncated(self) -> None:
        """text_delta excludes the stop string and text after it."""
        # Tokens: 0='A', 1='B', 2='C', 3='D'. Stop on "CD".
        # After decode with [0,1,2,3]: decoded="ABCD", stop at "CD".
        # The text_delta for the last step should only include text up to "CD".
        model = BatchedSequenceMockModel(vocab_size=100, sequences=[[0, 1, 2, 3]])
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [50], max_new_tokens=10, stop=["CD"])
        runner.prefill([req])
        # After prefill: generated=[0], text="A"

        runner.decode_step([req])
        # After decode 1: generated=[0,1], text="AB"

        outputs = runner.decode_step([req])
        # After decode 2: generated=[0,1,2], text="ABC", no stop yet
        assert outputs[0].finished is False

        outputs = runner.decode_step([req])
        # After decode 3: generated=[0,1,2,3], text="ABCD", stop at "CD"
        assert outputs[0].finished is True
        assert outputs[0].finish_reason == "stop"
        # text_delta should be empty since "AB" was already emitted and text truncates at "AB"
        assert "CD" not in outputs[0].text_delta


# ---------------------------------------------------------------------------
# Clear batch
# ---------------------------------------------------------------------------


class TestClearBatch:
    def test_clear_frees_state(self) -> None:
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3])
        runner.prefill([req])

        assert runner._kv_cache is not None
        assert runner._padding_mask is not None

        runner.clear_batch()

        assert runner._kv_cache is None
        assert runner._padding_mask is None
        assert runner._prompt_lens == []
        assert runner._max_prompt_len == 0
        assert runner._prev_text_lens == []


# ---------------------------------------------------------------------------
# StepOutput fields
# ---------------------------------------------------------------------------


class TestStepOutputFields:
    def test_non_finished_output_has_zero_counts(self) -> None:
        """Non-finished outputs have prompt_tokens=0 and completion_tokens=0."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        outputs = runner.prefill([req])

        assert outputs[0].finished is False
        assert outputs[0].prompt_tokens == 0
        assert outputs[0].completion_tokens == 0

    def test_finished_output_has_counts(self) -> None:
        """Finished outputs have correct prompt_tokens and completion_tokens."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=7)
        tokenizer = MockTokenizer()
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=1)
        outputs = runner.prefill([req])

        assert outputs[0].finished is True
        assert outputs[0].prompt_tokens == 3
        assert outputs[0].completion_tokens == 1

    def test_noop_output_for_finished_request(self) -> None:
        """Finished requests get noop outputs with token_id=None."""
        model = BatchedMockModel(vocab_size=100, fixed_next_token=99)
        tokenizer = MockTokenizer(eos_token_ids={99})
        config = _make_engine_config()
        runner = ModelRunner(model, tokenizer, config)  # type: ignore[arg-type]

        req = _make_request("r1", [1, 2, 3], max_new_tokens=5)
        runner.prefill([req])  # finishes on EOS

        outputs = runner.decode_step([req])
        assert outputs[0].token_id is None
        assert outputs[0].finished is True
        assert outputs[0].text_delta == ""
