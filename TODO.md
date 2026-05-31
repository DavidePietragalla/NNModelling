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
- [x] **convert.py integration test** — feed NNTree JSON → verify Hydra YAML structure

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

- [-] **Generate Hydra Test Configs** — run convert.py su subflow JSON, verify YAML
- [-] **Execute Dynamic Forward Pass** — main.py con config generati

### Phase 4: Exam Prep

- [ ] Review net/base.py line-by-line (topo sort, BFS, indice tracking)
- [ ] Sanitize auto-flatten spiegazione (leaky abstraction defense)

---

## 🆕 Aggiunte (non in TODO originale)

### Testing — Python side

- [x] **Test unit per ops Python** — 36 test in test_ops.py
- [x] **Test convert.py** — 35 test (parse_params, build_layer_config, subflow config, YAML)
- [x] **Test net/base.py** — 21 test (in_degrees, init dispatch, forward BFS, metric)
- [ ] **Config type checking** — mypy o pyright per converted/

### Docs & Config

- [ ] **pnpm-workspace.yaml** — untracked (`??`), da committare o rimuovere
- [ ] **CLAUDE.md missing stereotypes** — MaskedScaledDotProduct, PositionalEncoding, SequencePool non elencati
- [ ] **FlowCanvas.svelte:48** — remove dead `TODO` comment (`let isNodeSelected`)

### Infrastructure

- [ ] **CI pipeline (GitHub Actions)** — push → test frontend (vitest) + test backend (pytest). Due job separati: `frontend-ci` (node, pnpm, vitest) e `backend-ci` (python, uv, pytest). Serve anche lint (svelte-check, mypy)
- [ ] **Vitest integration tests per convert.py** — testare la pipeline completa: Diagram → NNTree → JSON → convert.py → Hydra YAML

### Training Pipeline — One-Button Train

- [ ] **FastAPI server (`converted/src/server.py`)** — endpoint `POST /train`, accetta NNTree JSON + params, lancia convert.py + main.py in subprocess, streama log
- [ ] **Redis job queue** — web server distaccato da cluster GPU. Redis come job broker (non cache). Pattern: LPUSH job, BRPOP worker, status polling via GET /status/:job_id, result in Redis
- [ ] **Worker process** — BRPOP da Redis, esegue convert + train, scrive risultato (pesi, log, metriche) su Redis/S3
- [ ] **Frontend "Convert & Train" button** — tasto in FlowCanvas toolbar (accanto a Save/Load/Convert). Chiama POST /train col NNTree JSON, mostra progresso (polling /status), notifica completamento
- [ ] **Vite proxy** — configurare `vite.config.ts` server.proxy per /train verso FastAPI backend
- [ ] **Artefatti training** — salvataggio pesi, log, metadati su S3/NFS accessibile da frontend per download

---

## Note

- `transformer_classifier.json` esiste (Svelte Flow format) ma mai testato E2E
- `HorizontalRepeat` join è hardcoded a concat su dim=-1 (`ops/horizontal_repeat.py:14`)
- `Concat` op testata in test_ops.py
