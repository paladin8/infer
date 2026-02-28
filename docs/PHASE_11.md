# Phase 11: Speculative Decoding

## Goal

Accelerate decode by using `meta-llama/Llama-3.2-1B-Instruct` as a draft model to propose multiple tokens per step, verified in a single forward pass of `meta-llama/Llama-3.2-3B-Instruct`. This is a lossless optimization — the output distribution is mathematically identical to pure target-model sampling.

Benchmark pair: `Llama-3.2-1B-Instruct` (draft) / `Llama-3.2-3B-Instruct` (target).

---

## Background

Autoregressive decode is inherently sequential — one forward pass per token. Each forward pass is memory-bandwidth-bound during decode (reading all model weights to produce a single token), so the GPU is underutilized. Speculative decoding exploits this by running a small draft model to produce K candidate tokens cheaply, then verifying all K candidates in a single forward pass of the target model.

The verification step uses rejection sampling to guarantee the output distribution is identical to pure target-model sampling. For a well-matched draft/target pair (same model family, shared vocabulary), acceptance rates of 70-90% are typical, yielding 1.5-2.5x effective decode throughput.

### Algorithm (Speculative Sampling)

Given a context sequence `x_1, ..., x_n`:

1. **Draft phase**: Run the draft model autoregressively for K steps, producing candidate tokens `t_1, ..., t_K` and their draft probabilities `q(t_i | x_1, ..., x_n, t_1, ..., t_{i-1})`.

2. **Verify phase**: Run the target model on the full sequence `x_1, ..., x_n, t_1, ..., t_K` in a single forward pass, obtaining target probabilities `p(t_i | ...)` for each candidate position, plus `p(t_{K+1} | ...)` for the position after the last candidate.

3. **Accept/reject**: For each candidate i = 1, ..., K:
   - Draw `r ~ Uniform(0, 1)`.
   - If `r < p(t_i) / q(t_i)`: **accept** token `t_i`, continue to next candidate.
   - Else: **reject** token `t_i`. Sample a correction token from `norm(max(0, p(·) - q(·)))` and stop — all subsequent candidates are discarded.

4. **Bonus token**: If all K candidates are accepted, sample one additional token from `p(t_{K+1} | ...)`. This means each speculation round produces between 1 and K+1 tokens.

5. **Greedy shortcut**: When `temperature=0`, acceptance simplifies to exact token match (`target_argmax == draft_token`). No rejection sampling needed — just compare tokens directly.

### Why It's Lossless

The rejection sampling criterion ensures that the marginal distribution of accepted tokens is exactly `p(t)`, the target model's distribution, regardless of the draft model's quality. Poor draft models just mean lower acceptance rates (fewer tokens per round), not different output.

### Why Llama 1B / 3B

- Same Llama 3.2 family, identical tokenizer and vocabulary (128,256 tokens).
- 1B model is ~3x cheaper per forward pass than 3B (fewer layers and smaller hidden dimension).
- Both fit comfortably in 16 GB VRAM: 1B ≈ 2.5 GB + 3B ≈ 6.5 GB = ~9 GB for weights, leaving ~7 GB for KV caches and activations.
- Same architecture means identical RoPE, attention pattern, and tokenization — no compatibility issues.

### Model Dimensions

|                       | Llama-3.2-1B | Llama-3.2-3B |
| --------------------- | ------------ | ------------ |
| Hidden size           | 2048         | 3072         |
| Intermediate size     | 8192         | 8192         |
| Layers                | 16           | 28           |
| Attention heads       | 32           | 24           |
| KV heads              | 8            | 8            |
| Head dim              | 64           | 128          |
| Vocab size            | 128,256      | 128,256      |
| Weight memory (bf16)  | ~2.5 GB      | ~6.5 GB      |

---

## Architecture

```text
Engine._step_continuous()
│
├── scheduler.retire() / scheduler.admit() / scheduler.decode_requests()
│   └── (unchanged — scheduler is unaware of speculative decoding)
│
└── runner.step(prefill, decode)
    │
    ├── Prefill: unchanged (_prefill_one, _prefill_batch, _prefill_chunks_batched)
    │   └── Only the target model prefills. Draft model needs no prefill —
    │       its KV cache is populated lazily during the first draft phase.
    │
    └── Decode: SpeculativeRunner._speculative_decode(requests)
        │
        ├── 1. Draft phase (K autoregressive steps on draft model)
        │   │
        │   │  For each step k = 1..K:
        │   │   ├── Build input_ids from last accepted/drafted token
        │   │   ├── Forward pass: draft_model(input_ids, kv_cache=draft_cache)
        │   │   ├── Sample draft token from draft logits
        │   │   └── Store draft_token and draft_log_prob
        │   │
        │   └── Result: K candidate tokens + K draft log-probs per request
        │
        ├── 2. Target verification (single forward pass)
        │   │
        │   │  Build input_ids = [last_accepted, t_1, ..., t_K] per request
        │   │  → shape [batch, K+1] (includes the token that triggered decode)
        │   │
        │   │  Forward pass: target_model(input_ids, kv_cache=target_cache)
        │   │  → target_logits: [batch, K+1, vocab_size]
        │   │  → Extract target log-probs for each candidate position
        │   │
        │   └── Result: K+1 sets of target log-probs per request
        │
        ├── 3. Accept/reject per request
        │   │
        │   │  For each request independently:
        │   │   ├── Greedy: accept while target_argmax == draft_token
        │   │   ├── Sampling: accept with prob min(1, p(t)/q(t))
        │   │   ├── On rejection: sample correction token from
        │   │   │   norm(max(0, p(·) - q(·)))
        │   │   ├── If all accepted: sample bonus token from p(t_{K+1})
        │   │   └── Record num_accepted for metrics
        │   │
        │   └── Result: 1..K+1 new tokens per request
        │
        └── 4. KV cache rollback
            │
            ├── Draft cache: roll back to last accepted position
            │   (discard KV entries for rejected draft tokens)
            │
            └── Target cache: roll back to last accepted position
                (the verification pass wrote K+1 entries but only
                num_accepted+1 are valid)
```

### Key Design Decisions

**Separate runner, not a mode within ContinuousRunner.** Speculative decoding fundamentally changes the decode path (multi-step draft + batched verify + rollback), and mixing this logic into `ContinuousRunner._batched_decode` would create an unwieldy method with interleaved control flow. A separate `SpeculativeRunner` class keeps the codepaths clean and testable. The `SpeculativeRunner` wraps a `ContinuousRunner` for prefill (reusing all existing prefill logic) and implements its own decode path.

**Draft model gets its own KV cache pool.** The draft model has different dimensions (hidden_size, num_kv_heads, head_dim, num_layers) than the target model, so it needs its own cache pool. Both pools use the same backend type (paged or contiguous) selected by `kv_cache_backend`. Draft cache slots are allocated and freed in lockstep with target cache slots — same slot indices, same lifecycle.

**Draft model prefill is lazy.** When a request is first admitted, only the target model runs prefill (populating the target KV cache). The draft model's KV cache for that request is populated during the first draft phase by running the draft model on the prompt. This avoids doubling prefill time for new requests. Implementation: the first draft phase detects an empty draft cache for a request and runs the draft model on the full prompt before starting the draft loop.

**Per-request speculation, not per-batch.** Each request has its own draft/verify cycle. Within a batch, all requests draft K tokens independently, then all are verified in a single batched target forward pass. Acceptance counts vary per request, so KV cache rollback is per-request.

**Batched draft and batched verification.** Draft generation is batched across all decode requests — each of the K draft steps is a batched forward pass on the draft model. Similarly, the verification pass is a single batched forward pass on the target model with input shape `[batch, K+1]`. This maximizes GPU utilization.

**Multi-token StepOutput.** Normal decode produces one `StepOutput` per request per step. Speculative decode can produce 1 to K+1 tokens per step. Rather than changing `StepOutput`, the runner emits multiple `StepOutput`s per request per step — one for each accepted token. `SpeculativeRunner.step()` returns `list[tuple[Request, StepOutput]]` where the same request may appear multiple times. The engine's `_step_continuous()` loop pushes all of them to the request's output queue in order. This is transparent to the API layer (it just reads tokens from the queue). Update the `StepOutput` docstring to note that speculative decoding may produce multiple outputs per request per step.

**Spec length is configurable but fixed per step.** The `spec_length` config (default 5) sets K for all requests in every round. Adaptive spec length (varying K based on per-request acceptance rate) is a future optimization.

---

## Deliverables

### D1: SpeculativeRunner class

Create `src/infer/engine/speculative_runner.py` with the `SpeculativeRunner` class:

- Constructor takes `target_model`, `draft_model`, `tokenizer`, and `config`.
- Creates two cache pools: `target_cache_pool` and `draft_cache_pool` (same backend as `config.kv_cache_backend`, but sized from respective model configs).
- Exposes the same interface as `ContinuousRunner`: `step(prefill, decode)`, `free_slot(slot_idx)`, `cleanup_request(request_id)`, `free_kv_tokens()`.
- `free_slot(slot_idx)` frees the slot in *both* pools. Slot indices are kept synchronized: when a target slot is allocated during prefill, a corresponding draft slot is allocated immediately (same `initial_tokens`). The `SpeculativeRunner` maintains a `_draft_slots: dict[int, int]` mapping target slot index to draft slot index, since the two pools may assign different internal IDs (especially for the paged backend where `seq_id` is auto-incremented).
- `free_kv_tokens()` returns `min(target_pool.free_token_capacity(), draft_pool.free_token_capacity())`, ensuring admission control accounts for both pools. If either pool returns `None` (contiguous backend), the other pool's value is used.
- `cleanup_request(request_id)` cleans up tracking state in both the target text tracking (`_prev_text_lens`) and the draft prefill set (`_draft_prefilled`), and removes the slot mapping from `_draft_slots`.
- Delegates prefill to the target model's cache pool and forward pass (same logic as `ContinuousRunner._prefill_one` / `_prefill_batch` / `_prefill_chunks_batched`). During prefill, also allocates the draft cache slot (but does NOT run draft prefill — that is deferred to the first draft phase).
- Implements `_speculative_decode(requests)` for the draft-then-verify loop.

The runner internally tracks per-request draft cache state via a `_draft_prefilled: set[str]` that records which request IDs have had their draft model KV cache populated.

**Files:** `src/infer/engine/speculative_runner.py`

### D2: Draft model loading

Extend `Engine.__init__` to load a second model when `use_speculative_decoding=True`:

1. Load the target model as before via `load_model(config.model, ...)`.
2. Load the draft model via `load_model(config.draft_model, ...)`.
3. Verify tokenizer compatibility: assert that `Tokenizer(config.model).vocab_size == Tokenizer(config.draft_model).vocab_size` and both have the same EOS token IDs.
4. Pass both models to `SpeculativeRunner`.

**Files:** `src/infer/engine/engine.py`

### D3: Draft generation loop

Implement `SpeculativeRunner._draft_generate(requests)`:

1. **Lazy prefill check**: Collect all requests not yet in `_draft_prefilled`. If any exist, run the draft model on their prompts to populate the draft KV cache using a batched prefill (same right-pad + padding_mask approach as `_prefill_batch`). This uses a `prefill_view` or `batched_prefill_view` on the draft cache pool — a regular prefill-style cache view, not a decode view. After the forward pass, mark all as prefilled in `_draft_prefilled`. The draft prefill output logits are discarded (the draft model's first real output comes from the draft loop below).

2. **Draft loop** (K iterations, where K = `config.spec_length`):
   - Build `input_ids` from each request's last token (accepted or previously drafted).
   - Batched forward pass on draft model: `draft_model(input_ids, kv_cache=draft_decode_view, ...)`.
   - Compute log-probabilities: `log_softmax(logits[:, -1, :])`.
   - Sample draft tokens (respecting each request's `SamplingParams` and `generator`).
   - Store `draft_tokens[req][k]` and `draft_log_probs[req][k]`.

3. Return per-request lists of draft tokens and draft log-prob tensors.

**Files:** `src/infer/engine/speculative_runner.py`

### D4: Target verification pass

Implement `SpeculativeRunner._verify(requests, draft_tokens)`:

1. **Input construction.** The target model's KV cache contains context through the last token that was *input* during the previous step. The most recently *sampled* token `g` (the last accepted token from the previous round, or the first generated token from prefill) has not yet been fed through the target model. Therefore, the verification input is `[g, t_1, ..., t_K]` — K+1 tokens per request, shape `[batch, K+1]`.

   The logits at each position give the target's distribution for the *next* token:
   - `target_logits[:, 0, :]` → target distribution for verifying `t_1` (conditioned on context through `g`)
   - `target_logits[:, k, :]` → target distribution for verifying `t_{k+1}` (conditioned on context through `t_k`)
   - `target_logits[:, K, :]` → bonus token distribution (conditioned on context through `t_K`)

2. **Cache view.** The verification pass is a "multi-token prefill" on the target model. Use `batched_chunked_prefill_view(slots, start_positions=[seq_len_per_req], chunk_lens=[K+1]*batch)` to write K+1 new entries per request. Provide `position_ids` (shape `[batch, K+1]`, values `[n, n+1, ..., n+K]` per request) and `padding_mask` (shape `[batch, n+K+1]`, True for valid positions) so the model constructs the correct per-element causal mask and RoPE embeddings. The verification view allocates blocks for all K+1 positions during the forward pass; `truncate_to` (D6) frees unused blocks after acceptance.

3. **Forward pass.** `target_model(input_ids, kv_cache=target_verify_view, padding_mask=padding_mask, position_ids=position_ids)` → `target_logits: [batch, K+1, vocab_size]`.

4. **Return** the target logits tensor for the acceptance step.

**Important**: The verification forward pass writes K+1 positions to the target KV cache (and calls `advance()` internally). After acceptance/rejection, the cache must be rolled back via `truncate_to` to only keep the accepted positions.

**Files:** `src/infer/engine/speculative_runner.py`

### D5: Acceptance and rejection sampling

Implement `SpeculativeRunner._accept_reject(requests, draft_tokens, draft_log_probs, target_logits)`:

For each request independently:

**Greedy mode** (`temperature == 0.0`):
- For each draft token `t_k`: if `argmax(target_logits[:, k-1, :]) == t_k`, accept; else reject.
- On rejection: the accepted token at this position is `argmax(target_logits[:, k-1, :])`.
- If all K accepted: bonus token is `argmax(target_logits[:, K, :])`.
- No randomness involved.

**Sampling mode** (`temperature > 0.0`):
- For each verification position k (0-indexed), construct the appropriate `context_token_ids` for sampling transforms: `prompt_tokens + generated_tokens + accepted_draft_tokens[:k]`. This means the repetition penalty context grows incrementally with each draft position, matching what the target model would see in normal autoregressive decode.
- Apply the request's full sampling pipeline (repetition penalty, temperature, top-k, top-p) to both draft and target logits at each position to get adjusted probability distributions. This is done per-position in a loop (K iterations per request), not vectorized, since `context_token_ids` differs at each position.
- Convert log-probabilities to probabilities via `exp()` for the acceptance computation. All arithmetic in the acceptance step uses probability space, not log space.
- For each draft token `t_k`:
  - Compute `r = p_target(t_k) / q_draft(t_k)` where probabilities are after sampling transforms.
  - Draw `u ~ Uniform(0, 1)` using the request's `generator`.
  - If `u < r`: accept.
  - Else: compute the correction distribution `p_corrected(·) = max(0, p_target(·) - q_draft(·))`, renormalize, sample a correction token from it, and stop.
- If all K accepted: sample bonus token from the adjusted `p_target(t_{K+1} | ...)`.

Return per-request: list of accepted token IDs, number accepted (for metrics).

**Files:** `src/infer/engine/speculative_runner.py`

### D6: KV cache rollback

After acceptance/rejection, roll back both draft and target KV caches to the correct position:

- At the start of the round, both caches are at `start_pos`. The draft phase feeds K tokens (starting with `g`), advancing the draft cache to `start_pos + K`. The verification phase feeds K+1 tokens (`[g, t_1, ..., t_K]`), advancing the target cache to `start_pos + K + 1`.
- If `num_accepted` draft tokens were accepted (0 ≤ num_accepted ≤ K), the valid cache entries are those for the input token `g` plus the `num_accepted` accepted draft tokens. The `+1` in the truncation target accounts for `g` — it was fed to both models at the start of this round and its KV entry must be kept.
- Draft cache: truncate from `start_pos + K` to `start_pos + num_accepted + 1`.
- Target cache: truncate from `start_pos + K + 1` to `start_pos + num_accepted + 1`.

Rollback is implemented by adjusting the sequence length counter in the cache pool. For the paged backend, this also means freeing any blocks that are now beyond the truncated length. For the contiguous/slotted backend, just decrementing `seq_len` suffices (stale entries are overwritten next round).

Add a `truncate_to(slot_idx, new_seq_len)` method to `CachePoolProtocol`:
- `SlottedKVCache`: set `slot_seq_lens[slot_idx] = new_seq_len`.
- `PagedKVCachePool`: set `seq_lens[slot_idx] = new_seq_len` and free blocks whose *first* token position is at or beyond `new_seq_len` (i.e., blocks with index `>= ceil(new_seq_len / block_size)`). Partial last blocks are kept — their stale entries are overwritten in the next round. Truncation only affects decode-phase blocks (tokens beyond the prompt); prefix tree blocks are never freed by `truncate_to` because draft/verification tokens are never inserted into the prefix tree.

**Prefix caching interaction**: `truncate_to` is safe with prefix caching because speculative tokens (draft and verification) are always beyond the prompt prefix. The prefix tree only contains prompt-prefix blocks, and `truncate_to` only frees blocks at positions beyond the truncation point. Prompt-prefix blocks are managed by the prefix tree's refcounting and are never touched by truncation.

**Files:** `src/infer/cache/protocol.py`, `src/infer/cache/slotted.py`, `src/infer/cache/paged.py`

### D7: Engine integration

Extend `Engine._init_components` to use `SpeculativeRunner` when `config.use_speculative_decoding`:

1. Load draft model: `draft_model, _ = load_model(config.draft_model, dtype=dtype, device=config.device)`. The draft model auto-detects its own quantization from its checkpoint (pass `quantization=None`), independent of the target model's `config.quantization` setting.
2. Verify tokenizer compatibility.
3. Create `SpeculativeRunner(target_model, draft_model, tokenizer, config)` instead of `ContinuousRunner`.
4. Update `_step_continuous()` to accept `SpeculativeRunner`: change the `isinstance` assert on `self.runner` to `isinstance(self.runner, (ContinuousRunner, SpeculativeRunner))` and update the type annotation on `self.runner` accordingly.

Extend `Engine.add_request` to validate that `prompt + max_new_tokens` fits within `max_seq_len` for both target and draft models (both use the same tokenizer, so prompt length is the same; `max_seq_len` governs both).

**Files:** `src/infer/engine/engine.py`

### D8: EngineConfig and CLI updates

Add speculative decoding fields to `EngineConfig`:

```python
# Speculative decoding (Phase 11)
use_speculative_decoding: bool = False
draft_model: str | None = None
spec_length: int = 5  # candidate tokens per speculation round
```

Validation rules:
- `use_speculative_decoding=True` requires `draft_model` to be set.
- `use_speculative_decoding=True` requires `batching_mode="continuous"`.
- `use_speculative_decoding=True` is incompatible with `use_cuda_graphs=True`.
- `spec_length` must be >= 1.

CLI arguments:
- `--speculative-decoding` (action `store_true`)
- `--draft-model` (str)
- `--spec-length` (int, default 5)

**Files:** `src/infer/engine/config.py`, `src/infer/server/__main__.py`

### D9: Acceptance rate metrics

Add per-request acceptance rate tracking:

- Add `speculation_acceptance_rates: list[float]` to `Request` (default empty list). After each speculation round, append `num_accepted / spec_length` to this list.
- At request completion, compute the average acceptance rate from the per-round list and include it in the final `StepOutput`.
- Add `acceptance_rate: float | None = None` to `StepOutput` (only populated on the final `done` event, computed from `Request.speculation_acceptance_rates`).

This enables tuning `spec_length` — if acceptance rates are consistently >90%, a larger K is beneficial; if <50%, a smaller K or different draft model is needed.

**Files:** `src/infer/engine/request.py`, `src/infer/engine/speculative_runner.py`

### D10: Tests

**Unit tests:**

- `test_accept_reject_greedy`: Verify that under greedy sampling, speculative decode produces exactly the same output as normal decode (exact token match on 5+ prompts).
- `test_accept_reject_sampling`: Verify rejection sampling correctness — construct known draft/target probability distributions, verify acceptance probabilities match the theoretical criterion.
- `test_kv_cache_rollback`: Verify `truncate_to` correctly adjusts sequence length and frees blocks (paged backend).
- `test_draft_target_tokenizer_mismatch`: Verify that loading incompatible draft/target models raises a clear error.
- `test_config_validation`: Verify `use_speculative_decoding=True` without `draft_model` raises `ValueError`.
- `test_multi_token_output`: Verify that a single speculation round can produce multiple `StepOutput`s per request.

**Integration tests (GPU, requires models):**

- `test_speculative_greedy_parity`: Load Llama 1B/3B, run greedy decode with and without speculation, verify identical output for 3+ prompts.
- `test_speculative_e2e`: Load Llama 1B/3B, run sampling decode, verify output is coherent and acceptance rate is logged.
- `test_speculative_with_continuous_batching`: Multiple concurrent requests, some speculative, verify all complete correctly.

**Files:** `tests/unit/test_speculative.py`, `tests/integration/test_speculative_e2e.py`

---

## Implementation Details

### Verification Forward Pass

The verification pass is essentially a "multi-token prefill" on the target model — it processes K+1 tokens for each request. The KV cache view must handle writing K+1 new positions. This is identical to the existing chunked prefill logic — we can reuse `batched_chunked_prefill_view` with `start_position = current_seq_len` and `chunk_len = K+1`.

The target model's causal mask must be correct: each verification position can attend to all prior context plus all verification positions up to and including itself. The existing position_ids-based mask construction in the Llama forward pass handles this — we pass `position_ids = [[n, n+1, ..., n+K]]` and the mask is built from position comparisons.

### Draft Model KV Cache Sizing

The draft model's KV cache uses the same `max_batch_size` and `max_seq_len` as the target model. Since the draft model has fewer layers and smaller head dimensions, its KV cache is much smaller:

```
Draft KV (1B):  2 * 16 layers * 8 kv_heads * 64 head_dim * max_tokens * 2 bytes
              = 32,768 * max_tokens bytes per sequence

Target KV (3B): 2 * 28 layers * 8 kv_heads * 128 head_dim * max_tokens * 2 bytes
              = 114,688 * max_tokens bytes per sequence
```

At `max_seq_len=4096`:
- Draft KV per sequence: ~128 MB → 8 sequences ≈ 1 GB
- Target KV per sequence: ~448 MB → 8 sequences ≈ 3.5 GB

Total VRAM budget: ~2.5 GB (1B weights) + ~6.5 GB (3B weights) + ~1 GB (draft KV) + ~3.5 GB (target KV) ≈ ~13.5 GB. Fits within 16 GB with room for activations.

### Sequence Length Overflow Guard

During a speculation round, the draft phase writes K entries and the verification phase writes K+1 entries to the KV caches. If a request's `current_seq_len + K + 1 > max_seq_len`, the cache would overflow. Before each speculation round, the runner checks each request's remaining capacity. If `remaining = max_seq_len - current_seq_len` is less than `K + 1`, the runner reduces the effective spec length for that request to `max(remaining - 1, 0)`. If `remaining <= 1`, the request falls back to normal single-token decode for this step (feed the last token to the target model directly, sample one token, no draft/verify cycle).

### Handling Variable Acceptance Across Batch

Different requests in the batch may accept different numbers of tokens. After the accept/reject step:

1. Roll back each request's KV caches independently (per-slot truncation).
2. Emit the appropriate number of `StepOutput`s per request.
3. The next speculation round starts fresh — all requests draft K new tokens from their respective positions.

No special padding is needed because each speculation round is self-contained.

### Draft Log-Probability Computation

During the draft phase, we need log-probabilities for the sampled tokens. For greedy mode, this is unnecessary (acceptance is just token comparison). For sampling mode:

```python
draft_logits = draft_model(input_ids, kv_cache=draft_view)
# Apply same sampling transforms as target will use
adjusted_logits = apply_sampling_transforms(draft_logits, params)
draft_log_probs = log_softmax(adjusted_logits)
draft_token_log_prob = draft_log_probs[sampled_token]
```

We store the full draft log-probability distribution (not just the sampled token's log-prob) because the rejection sampling correction step needs to compute `max(0, p(·) - q(·))` over the full vocabulary. Log-probs are stored during drafting; they are converted to probabilities via `exp()` only at acceptance time. The correction distribution `max(0, p_target - q_draft)` is computed in probability space (not log space) and renormalized before sampling. In practice, storing the full `[vocab_size]` log-prob tensor per draft step is feasible (128K * 4 bytes * K steps * batch_size ≈ a few MB).

### Interaction with Existing Features

- **Prefix caching**: Works transparently. The target model's prefill uses prefix caching as before. The draft model's lazy prefill does not use prefix caching (the draft model's KV is cheap enough that this is fine).
- **Chunked prefill**: Works transparently. Speculative decoding only affects the decode path; prefill is unchanged.
- **Quantization**: Both draft and target models can be quantized independently. The `quantization` config applies to the target model; the draft model uses whatever quantization is auto-detected from its checkpoint.
- **CUDA graphs**: Not compatible with speculative decoding (the verification pass has variable-length input, not `seq_len=1`). Config validation should reject `use_cuda_graphs=True` with `use_speculative_decoding=True`.
- **Stop conditions**: The draft phase does *not* check stop conditions — it always runs for all K steps regardless of whether draft tokens include EOS or stop strings. This keeps the draft loop simple and branchless. Stop conditions are checked during the accept/reject phase on accepted tokens only. If a stop condition triggers mid-speculation (e.g., accepted token 3 of 5 is EOS), the remaining tokens are discarded and the request finishes immediately.

---

## VRAM Budget

| Component                              | Size          |
| -------------------------------------- | ------------- |
| Target model weights (3B, bf16)        | ~6.5 GB       |
| Draft model weights (1B, bf16)         | ~2.5 GB       |
| Target KV cache (8 seq × 4096 tokens)  | ~3.5 GB       |
| Draft KV cache (8 seq × 4096 tokens)   | ~1.0 GB       |
| Activations + temporaries              | ~1.5 GB       |
| **Total**                              | **~15.0 GB**  |

Fits within 16 GB dev tier. For larger batch sizes or longer sequences, reduce `max_batch_size` or `max_seq_len`, or use quantized models.

---

## Risks and Mitigations

1. **Low acceptance rate.** If the 1B draft model diverges too much from the 3B target, acceptance rates drop below 50% and speculative decoding becomes slower than normal decode (K cheap draft passes + 1 expensive verify pass to get <K/2 tokens). Mitigation: log acceptance rates, make `spec_length` tunable, document expected rates. If rates are consistently low, the user can disable speculation.

2. **VRAM pressure from two models.** Hosting both models simultaneously roughly triples weight memory vs target-only. Mitigation: Llama 1B is small enough (~2.5 GB) that this is manageable on 16 GB. For larger target models, the draft model can be quantized (FP8/INT8) to halve its weight memory.

3. **Draft prefill latency spike.** The first speculation round for each request runs the draft model on the full prompt (lazy prefill). For long prompts, this adds noticeable latency to the first decode step. Mitigation: the draft model is 3x faster than the target for prefill, so this is bounded. For very long prompts, the draft prefill is still <1/3 of the target prefill time. Could be optimized in future by running draft prefill concurrently with target prefill.

4. **Multi-token StepOutput ordering.** Emitting multiple StepOutputs per request per step requires the API layer to handle bursts of tokens. Mitigation: the existing `asyncio.Queue`-based output channel handles this naturally — tokens are pushed in order and the SSE layer reads them sequentially.

5. **KV cache rollback correctness for paged backend.** Freeing blocks during truncation must correctly handle partial blocks (last block may contain valid entries beyond the truncation point). Mitigation: `truncate_to` only frees blocks whose *starting* position is beyond the new length; partial last blocks are kept and their stale entries are overwritten in the next round.

---

## Exit Criteria

- Measurable tokens-per-second improvement on single-request decode with Llama 1B/3B pair (expect 1.5-2.5x depending on acceptance rate).
- Correctness: greedy decode output is identical with and without speculation (exact token match on 5+ diverse prompts).
- Correctness: sampling mode output is statistically equivalent (verified via fixed-seed deterministic comparison and distribution tests).
- Acceptance rate logged per request at completion.
- No regression when `use_speculative_decoding=False` (default).
- All existing tests pass with speculation disabled.
- Benchmark comparing speculative vs non-speculative decode on the W1 (single request) workload.
