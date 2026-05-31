# Todo List for NNModelling — Stato Attuale

## Overview

Stato aggiornato al 2026-05-31. Basato su TODO.md originale + audit effettivo del codice.

Legenda: `[x]` done, `[ ]` pending, `[-]` partially done

---

### Phase 1: Test Setup & Base Hardening

- [x] Configurare Vitest
- [x] Ground-truth snapshot regression test
- [x] Core compiler test (nnTree.ts)
- [x] Loop detection test (checkValidConnection)
- [x] Fix svelte-check type errors
- [x] Refactor tests to use real Diagram

### Phase 1b: Test Coverage Gaps

- [x] Self-loop edge case test
- [x] SubFlow compilation test (auto_encoder_submodels.json)
- [ ] **convert.py integration test** — feed NNTree JSON → verify Hydra YAML structure

### Phase 2: SubFlow Stereotype Conversion

- [x] Write failing SubFlow test spec (13 tests TDD)
- [x] Extend nnTree.ts (isSubflowNode, compileSubflowLayers, unrollLayers, etc.)
- [-] **convert.py subflow handling** — base subflow config esiste (`_build_nested_subflow_config`), ma mancano:
  - `iterations` param per Repeat/HorizontalRepeat (stereotype comportamentale)
  - Preservazione metadata stereotipo nei nodi subflow

### Phase 2b: Subflow Fixes — Nested & Metadata

- [ ] **Nested subflow test** — test + fix per subflow dentro subflow ricorsivo
- [ ] **Stereotype metadata in output** — preservare `stereotype`, `iterations`, params in NNTreeNode per subflow nodes
- [ ] **Test subflow senza unrolling** — plain container (no stereotype) skipa Iterations

### Phase 2d: New Ops for Transformer

- [x] MaskedScaledDotProduct Join op
- [x] PositionalEncoding Module op
- [x] SequencePool Module op
- [x] Transformer classifier diagram (transformer_classifier.json)
- [ ] **`join_type` param per HorizontalRepeat** — supportare concat/add/mean/max/mul
- [ ] **E2E training test** — convert.py → main.py su EnronSpam/transformer_classifier

### Phase 2f: Join Input Ordering

- [x] Join ordering fix (targetHandle preservation)
- [x] Remove auto-flatten in base.py
- [x] Update all diagram JSONs

### Phase 3: E2E Integration & Verification

- [ ] **Generate Hydra Test Configs** — run convert.py su subflow JSON, verify YAML
- [ ] **Execute Dynamic Forward Pass** — main.py con config generati

### Phase 4: Exam Prep

- [ ] Review net/base.py line-by-line (topo sort, BFS, indice tracking)
- [ ] Sanitize auto-flatten spiegazione (leaky abstraction defense)

---

## 🆕 Aggiunte (non in TODO originale)

### Testing — Python side (ZERO coverage oggi)

- [ ] **Test unit per ops Python** — `ops/addition.py`, `einsum.py`, `concat.py`, `mat_mul.py`, `scaled_dot_product.py`, `masked_scaled_dot_product.py`, `positional_encoding.py`, `sequence_pool.py`, `subflow.py`, `repeat.py`, `horizontal_repeat.py` — nessun test esiste
- [ ] **Test convert.py** — parsing params, generazione YAML, gestione errori
- [ ] **Test net/base.py** — forward pass, topo sort, join dispatch, subflow execution
- [ ] **Config type checking** — mypy o pyright per converted/

### Docs & Config

- [ ] **pnpm-workspace.yaml** — untracked (`??`), da committare o rimuovere
- [ ] **CLAUDE.md missing stereotypes** — MaskedScaledDotProduct, PositionalEncoding, SequencePool non elencati
- [ ] **FlowCanvas.svelte:48** — remove dead `TODO` comment (`let isNodeSelected`)

### Infrastructure

- [ ] **Vitest integration tests per convert.py** — testare la pipeline completa: Diagram → NNTree → JSON → convert.py → Hydra YAML

---

## Note

- `transformer_classifier.json` esiste (Svelte Flow format) ma mai testato E2E
- `HorizontalRepeat` join è hardcoded a concat su dim=-1 (`ops/horizontal_repeat.py:14`)
- `Concat` op esiste ma zero test
