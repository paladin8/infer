# Project Memory

## Current Phase: Phase 11 — Speculative Decoding
- Status: COMPLETE
- Design doc: `docs/PHASE_11.md`
- Draft/target pair: Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct

## Completed Phases
- Phases 1-10: All complete (see OVERALL_DESIGN.md)
- Phase 10: INT8 per-channel quantization support
- Phase 11: Speculative decoding with draft/target model pair

## Phase 11 Deliverables (all DONE)
1. D8: EngineConfig + CLI updates (foundation)
2. D6: KV cache truncate_to (foundation)
3. D9: Acceptance rate metrics (foundation)
4. D1+D3+D4+D5: SpeculativeRunner (core)
5. D7: Engine integration
6. D10: Unit tests (38 tests) + integration tests (3 GPU tests)

## Phase 11 Key Implementation Notes
- SpeculativeRunner: `src/infer/engine/speculative_runner.py` (~1100 lines)
- Dual KV cache pools (target + draft) with synchronized slot lifecycle
- Lazy draft prefill on first decode step (moved to _speculative_decode for correct rollback)
- Draft cache rollback capped at K (draft runs K passes, target runs K+1)
- Greedy mode: exact token match acceptance
- Sampling mode: rejection sampling with correction distribution
- Unit test count: 1047 (up from 1002 baseline)
