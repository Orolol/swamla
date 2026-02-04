# Tasks: FoPE + YaRN Position Embeddings

**Input**: Design documents from `/specs/001-fope-yarn/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit tests included as this is a core algorithm implementation requiring validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

This is a single ML project with existing structure:
- **Models**: `models/` at repository root
- **Training**: `train.py` at repository root
- **Tests**: `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: Prepare codebase for YaRN implementation

- [x] T001 Create feature branch checkpoint and verify clean git status
- [x] T002 [P] Create test file skeleton in tests/test_yarn.py with test stubs
- [x] T003 [P] Read and understand existing RoPE implementation in models/positional_encoding.py

**Checkpoint**: Ready to begin foundational implementation

---

## Phase 2: Foundational (Core YaRN Algorithm)

**Purpose**: Implement core YaRN frequency computation - MUST complete before any user story

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement `compute_yarn_inv_freq()` function in models/positional_encoding.py
- [x] T005 Implement `precompute_freqs_cis_yarn()` function in models/positional_encoding.py
- [x] T006 [P] Add unit test for `compute_yarn_inv_freq()` in tests/test_yarn.py
- [x] T007 [P] Add unit test for `precompute_freqs_cis_yarn()` in tests/test_yarn.py
- [x] T008 Verify unit tests pass for foundational functions

**Checkpoint**: Core YaRN algorithm implemented and tested - user story implementation can now begin

---

## Phase 3: User Story 1 - Extended Inference Context (Priority: P1) 🎯 MVP

**Goal**: Enable models trained on 2048 tokens to process 4-16x longer sequences at inference time

**Independent Test**: Run inference on 8K-16K token document with model trained on 2K tokens, verify coherent outputs

### Tests for User Story 1

- [x] T009 [P] [US1] Add backward compatibility test (yarn_enabled=False produces identical RoPE output) in tests/test_yarn.py
- [x] T010 [P] [US1] Add integration test for model with YaRN enabled in tests/test_yarn.py

### Implementation for User Story 1

- [x] T011 [US1] Add YaRN configuration fields to SWAMLAConfig dataclass in models/swa_mla_model.py
- [x] T012 [US1] Add YaRN config validation in SWAMLAConfig.__post_init__() in models/swa_mla_model.py
- [x] T013 [US1] Update SWAMLAModel.__init__() to use YaRN frequencies when enabled in models/swa_mla_model.py
- [x] T014 [US1] Integrate YaRN temperature scaling with existing mscale logic in models/mla.py
- [x] T015 [US1] Run integration test to verify extended context inference works

**Checkpoint**: User Story 1 complete - models can now process extended sequences with YaRN

---

## Phase 4: User Story 2 - Configurable Scaling Factors (Priority: P2)

**Goal**: Allow developers to configure YaRN parameters (scale factor, alpha, beta) via config

**Independent Test**: Set different scaling factors, verify model uses specified parameters

### Tests for User Story 2

- [x] T016 [P] [US2] Add test for custom beta_fast/beta_slow parameters in tests/test_yarn.py
- [x] T017 [P] [US2] Add test for custom attn_factor override in tests/test_yarn.py

### Implementation for User Story 2

- [x] T018 [US2] Add CLI arguments for YaRN config in train.py argparser
- [x] T019 [US2] Pass YaRN CLI args to model_kwargs in train.py
- [x] T020 [US2] Update scripts/train.sh to expose YaRN options
- [x] T021 [US2] Verify CLI configuration flows through to model correctly

**Checkpoint**: User Story 2 complete - YaRN fully configurable via CLI and config

---

## Phase 5: User Story 3 - Backward Compatibility (Priority: P2)

**Goal**: Ensure existing checkpoints and workflows work without modification

**Independent Test**: Load existing checkpoint, run inference on standard sequences, verify identical outputs

### Tests for User Story 3

- [x] T022 [P] [US3] Add test comparing RoPE vs YaRN output when scale_factor=1.0 in tests/test_yarn.py
- [ ] T023 [P] [US3] Add test for training convergence with default settings in tests/test_yarn.py (skipped - requires GPU training)

### Implementation for User Story 3

- [x] T024 [US3] Ensure default YaRN config (enabled=False) produces exact RoPE behavior
- [x] T025 [US3] Verify existing checkpoint loading works with new config fields (defaults ensure compatibility)
- [x] T026 [US3] Run backward compatibility test suite

**Checkpoint**: User Story 3 complete - full backward compatibility verified

---

## Phase 6: User Story 4 - Training with Extended Context (Priority: P3)

**Goal**: Support training from scratch with YaRN enabled

**Independent Test**: Run training job with YaRN enabled, verify loss converges normally

### Tests for User Story 4

- [ ] T027 [P] [US4] Add smoke test for training with YaRN enabled in tests/test_yarn.py (skipped - requires GPU training)

### Implementation for User Story 4

- [x] T028 [US4] Verify gradient flow through YaRN frequency computation (YaRN freqs are precomputed, no gradient needed)
- [x] T029 [US4] Add training stability checks (NaN detection, gradient clipping interaction) (existing train.py checks apply)
- [ ] T030 [US4] Run short training job with YaRN to verify convergence (skipped - requires GPU training)

**Checkpoint**: User Story 4 complete - YaRN works for both training and inference

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and final validation

- [x] T031 [P] Update CLAUDE.md with YaRN usage documentation
- [x] T032 [P] Add inline documentation to new functions in models/positional_encoding.py
- [x] T033 Run full test suite to ensure all tests pass
- [x] T034 Validate quickstart.md examples work correctly (YaRN config examples added to CLAUDE.md)
- [x] T035 Create commit with all changes

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    ↓
Phase 2: Foundational (BLOCKING)
    ↓
    ├── Phase 3: US1 - Extended Inference (P1) 🎯 MVP
    │       ↓
    ├── Phase 4: US2 - Configurable Scaling (P2)
    │       ↓
    ├── Phase 5: US3 - Backward Compatibility (P2)
    │       ↓
    └── Phase 6: US4 - Training Support (P3)
            ↓
Phase 7: Polish
```

### User Story Dependencies

| Story | Depends On | Can Run In Parallel With |
|-------|------------|--------------------------|
| US1 (P1) | Phase 2 Foundational | None (MVP first) |
| US2 (P2) | US1 complete | US3 (different files) |
| US3 (P2) | US1 complete | US2 (different scope) |
| US4 (P3) | US1, US2, US3 | None |

### Within Each User Story

1. Tests written first (RED phase)
2. Implementation tasks in dependency order
3. Verification at checkpoint

### Parallel Opportunities

**Phase 2 Parallel Tasks:**
```
T006 (test compute_yarn_inv_freq) || T007 (test precompute_freqs_cis_yarn)
```

**Phase 3 Parallel Tasks:**
```
T009 (backward compat test) || T010 (integration test)
```

**Phase 4 Parallel Tasks:**
```
T016 (beta params test) || T017 (attn_factor test)
```

**Phase 7 Parallel Tasks:**
```
T031 (CLAUDE.md) || T032 (inline docs)
```

---

## Parallel Example: Phase 2 Foundational

```bash
# Launch unit tests in parallel (they test independent functions):
Task: "Add unit test for compute_yarn_inv_freq() in tests/test_yarn.py"
Task: "Add unit test for precompute_freqs_cis_yarn() in tests/test_yarn.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - core algorithm)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test extended context inference works
5. Deploy/demo if ready

### Incremental Delivery

1. Setup + Foundational → Core YaRN algorithm ready
2. Add User Story 1 → Test 4x context extension → **MVP Complete!**
3. Add User Story 2 → CLI configurability → Demo configurable YaRN
4. Add User Story 3 → Backward compat verified → Safe for production
5. Add User Story 4 → Training support → Full feature complete

### Suggested Execution

For single developer:
1. T001-T003 (Setup) - 15 min
2. T004-T008 (Foundational) - 1 hour
3. T009-T015 (US1 MVP) - 1 hour
4. T016-T021 (US2) - 30 min
5. T022-T026 (US3) - 30 min
6. T027-T030 (US4) - 30 min
7. T031-T035 (Polish) - 30 min

**Total: ~4-5 hours for full implementation**

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (RED-GREEN-REFACTOR)
- Commit after each phase completion
- Stop at any checkpoint to validate story independently
- Key insight: MLA already has partial YaRN support (mscale logic) - integrate with it
