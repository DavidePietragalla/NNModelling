# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**NNModelling** — DSL for designing neural networks via visual node editor. Diagrams convert to PyTorch/Lightning code.

Two main parts:

1. **front-end/** — Svelte 5 + Svelte Flow visual editor (TypeScript)
2. **converted/** — Python codegen target (PyTorch + Lightning + Hydra)

### Tech Stack

- **Frontend**: Svelte 5, Svelte Flow (@xyflow/svelte), Vite 8, TypeScript
- **Build**: pnpm v10
- **Python**: PyTorch, Lightning (LightningModule), Hydra (config), wandb (logging)
- **Vitest** (front-end unit tests, configured in `vitest.config.ts`)
- **pytest** (Python unit tests, in `converted/src/tests/`)

### Workflow

1. Drag module nodes from sidebar onto canvas
2. Connect nodes to define data flow
3. Configure parameters via sidebar (position, color, params)
4. Save/Load diagrams as JSON (`.json` files)
5. Convert to Python: Diagram → NNTree → JSON → `convert.py` → Hydra YAML configs

## Dev Commands

```bash
cd front-end
npm run dev       # Start Vite dev server
npm run build     # Production build
npm run preview   # Preview built app
npm run check     # Type-check Svelte/TS (svelte-check)
npm run test      # Run unit tests (vitest, single run)
npm run test:watch  # Run unit tests (vitest, watch mode)
npm run test:integration           # Run integration tests (all tiers)
npm run test:integration:smoke     # Tier 0: NNTree compilation only
npm run test:integration:convert   # Tier 1: convert.py YAML generation
npm run test:integration:forward   # Tier 2: Net.forward() pass tests
npm run test:integration:train     # Tier 3: main.py training smoke (slow, CPU/GPU)
npm run test:integration:infer     # Tier 4: infer.py output validation
npm run test:integration:all       # All integration tiers
NNM_DIAGRAM=mninst npm run test:example  # Single diagram through all tiers
NNM_DEVICE=gpu NNM_DIAGRAM=mninst npm run test:integration:train  # GPU training
```

```bash
cd converted
uv run python src/convert.py <nn_tree_json> <output_dir>  # Generate Hydra configs
uv run python src/main.py --config-dir <dir>               # Train model
uv run python src/infer.py --config-path <dir> --config-name <name> --weights <path>  # Inference
```

## Project Structure

```
NNModelling/
├── front-end/src/              # Svelte 5 visual editor
│   ├── __tests__/              # Vitest test suites
│   │   ├── helpers.ts          # Test factories (stubWindow, node, edge)
│   │   ├── nnTree.test.ts      # NNTree regression tests
│   │   ├── typeEngine.test.ts  # Type inference unit tests
│   │   ├── utils.test.ts       # checkValidConnection tests
│   │   └── integration/        # Integration test suite (tiered, Python pipeline)
│   │       ├── helpers.ts      # Shared helpers (manifest, uvRun, pipeline stages)
│   │       ├── smoke.test.ts   # Tier 0: nnTree compilation
│   │       ├── convert.test.ts # Tier 1: convert.py YAML generation
│   │       ├── forward.test.ts # Tier 2: Net.forward() pass
│   │       ├── train.test.ts   # Tier 3: main.py training smoke
│   │       └── infer.test.ts   # Tier 4: infer.py output validation
│   ├── nodes/
│   │   ├── CustomNode.svelte   # Standard NN module node (with type error indicator badges)
│   │   ├── JoinNode.svelte     # Merge node (multi-input) (with type error indicator badges)
│   │   └── SubflowNode.svelte  # Collapsible submodel container (with type error indicator badges)
│   ├── conversion/
│   │   ├── nnTree.ts           # Diagram → tree representation
│   │   ├── tensortypes.ts      # Tensor type model interfaces
│   │   └── typeEngine.ts       # Constraint-based type inference engine
│   ├── styles/                 # CSS (flowcanvas, node, sidebar, join, subflow, dropdown)
│   ├── components/
│   │   ├── Sidebar.svelte      # Node create/edit form
│   │   └── SDropdown.svelte    # Custom dropdown widget
│   ├── Diagram.svelte.ts       # Central reactive state manager
│   ├── FlowCanvas.svelte       # Main editor canvas + toolbar
│   ├── stereotype.ts           # Stereotype loader (import.meta.glob)
│   ├── utils.ts                # Connection validation, helpers
│   ├── App.svelte              # Entry point (SvelteFlowProvider)
│   └── main.ts                 # Mount point
├── Stereotypes/                # JSON template definitions
│   ├── Modules/                # 27 layers (Input, Linear, Conv2d, ReLU, etc.)
│   ├── Joins/                  # Addition, Concat, Einsum, MatMul, ScaledDotProduct, MaskedScaledDotProduct
│   └── SubFlows/               # Repeat, HorizontalRepeat templates
├── examples/                   # Test fixtures for integration tests
│   ├── manifest.json           # Diagram metadata (input shapes, task type, trainable flags)
│   ├── diagrams/               # Svelte Flow format source diagrams
│   └── nntrees/                # Pre-compiled NNTree JSON files
├── converted/src/              # Python codegen target
│   ├── net/base.py             # LightningModule: dynamic ModuleDict + topo sort
│   ├── ops/                    # Custom operations
│   │   ├── addition.py         # Add join operation
│   │   ├── einsum.py           # Einsum join operation
│   │   ├── concat.py           # Concat join operation
│   │   ├── horizontal_repeat.py  # HorizontalRepeat: N parallel copies via vmap
│   │   ├── mat_mul.py          # MatMul operation
│  │   │   ├── masked_scaled_dot_product.py  # Masked scaled dot product attention
│   │   ├── mat_mul.py          # MatMul operation
│   │   ├── positional_encoding.py  # PositionalEncoding operation
│   │   ├── scaled_dot_product.py  # Scaled dot product attention
│   │   ├── sequence_pool.py    # SequencePool operation
│   │   ├── subflow.py          # Subflow: BFS execution of internal graph
│   │   └── repeat.py           # Repeat: N copies of subgraph in Sequential
│   ├── dataset/mnist.py        # MNIST dataset class
│   ├── dataset/autoencoder_mnist.py  # Autoencoder MNIST dataset
│   ├── dataset/enron_spam.py   # EnronSpam text classification dataset (HF datasets + transformers)
│   ├── convert.py              # NNTree JSON → Hydra config dir
│   ├── main.py                 # Training entry point (Hydra + Lightning)
│   ├── infer.py                # Inference on trained model (--output, --image-dir)
│   └── tests/                  # Python test suite (pytest)
│       ├── test_ops.py         # 36 ops unit tests
│       ├── test_convert.py     # 35 convert.py unit tests
│       ├── test_base.py        # 21 Net/base unit tests
│       ├── test_integration.py # 11 end-to-end integration tests
│       ├── test_main.py        # 2 training smoke tests
│       └── test_infer.py       # 4 inference validation tests
├── analysis/
│   ├── requirements/reqs.md    # DSL requirements specification
│   └── uml/nn.vpp              # UML model (Visual Paradigm)
├── *.json                      # Example diagram files
│   ├── mninst.json             # Simple MNIST classifier
│   ├── mnist_skips.json        # MNIST with skip connections
│   ├── autoencoder_mnist.json  # Autoencoder diagram
│   ├── auto_encoder_submodels.json  # Autoencoder with submodels
│   ├── single_head_attention.json  # Single-head attention from primitives
│   ├── skip_connections_with_repetition.json  # Residual + Repeat subflow
│   ├── multihead_attention.json  # 4-head attention via Concat join
│   ├── horizontal_multihead_attention.json  # 4-head attention via HorizontalRepeat
│   ├── transformer_classifier.json  # Full transformer: embed, posenc, encoder×2, pool, linear
│   └── .qwen/                      # Qwen AI settings (legacy)
```

## Architecture

### Testing — Front-end (Vitest Unit Tests)

- **Framework**: Vitest (v4), configured in `vitest.config.ts`
- **Pattern**: Pure TS unit tests, no DOM/browser
- **Real Diagram**: Tests use real `Diagram` class (Svelte `$state.raw` compiled by Vite plugin). Stub `globalThis.window` before construction.
- **Helpers**: `node(id, stereo, name, params, overrides?)` and `edge(id, source, target, handles?)` for concise fixtures.
- **Coverage**: **102 tests** — sequential chain, skip/joins, autoencoder, subflow compilation, nested subflows, hidden nodes, error handling, connection validation, Fork node, type inference (Input, Linear, ReLU, Conv2d, Flatten, MaxPool2d, wildcards, mismatches, edge cases, computed dimensions, join type checking)

### Testing — Integration (Vitest + Python Pipeline)

- **Config**: `vitest.integration.config.ts` (separate from unit tests), uses `pool: "forks"` for isolation, 10min timeout
- **Pattern**: TypeScript orchestration via Node.js `child_process` spawning `uv run python ...` commands
- **Manifest**: `examples/manifest.json` drives which diagrams to test — defines `inputShape`, `taskType`, `trainable` flags for each diagram
- **Tiered execution**: Controlled via `NNM_TIER` env var; tests skip when tier doesn't match

| Tier | Env Value | Test File | What It Validates |
|------|-----------|-----------|-------------------|
| 0 | `smoke` | `smoke.test.ts` | NNTree compilation from Svelte Flow JSON |
| 1 | `convert` | `convert.test.ts` | `convert.py` YAML generation (expected files, config structure) |
| 2 | `forward` | `forward.test.ts` | `Net.forward()` pass — loads config, runs forward with mock tensors |
| 3 | `train` | `train.test.ts` | `main.py` training smoke test (1 epoch, validated by exit code + checkpoint) |
| 4 | `infer` | `infer.test.ts` | `infer.py` output validation (JSON predictions, image output) |

- **Env vars**: `NNM_DIAGRAM` (single diagram), `NNM_DEVICE` (cpu/gpu), `NNM_TIER` (tier filter), `NNM_WANDB_MODE` (disable by default), `NNM_KEEP_TEMP` (preserve temp dirs for debugging)

### Testing — Python (pytest)

- **Framework**: pytest (via `uv run pytest`)
- **Files**: `converted/src/tests/` — pure Python tests (no TS/Vitest dependency)
- **Coverage**: **103 Python unit tests** across `test_ops.py` (36), `test_convert.py` (35), `test_base.py` (21), `test_integration.py` (11); **6 Python pipeline tests** in `test_main.py` (2) and `test_infer.py` (4)
- **Fixtures**: NNTree JSON files from `examples/nntrees/` — shared between Python and Vitest integration tests

### Front-end Data Flow

```
Stereotypes/ (JSON) → Stereotype class → Diagram class (reactive state) → FlowCanvas.svelte (Svelte Flow UI)
                                                                               ↓
                                                                    TypeEngine (type checking)
                                                                               ↓
                                                                          NNTree class (conversion)
                                                                               ↓
                                                                          JSON → convert.py (Hydra configs)
```

### Key Classes

- **Diagram.svelte.ts** — Central state manager. Holds `nodes` and `edges` as Svelte 5 `$state.raw` arrays. Methods: `addModule`, `addJoinNode`, `addSubGraph`, `deleteNodes`, `importFromJson`, `exportToJson`, `toggleSubflow`.
- **Stereotype** (stereotype.ts) — Loads all JSON files from `Stereotypes/**/*.json` via Vite's `import.meta.glob`. Each stereotype defines: `pythonClassName`, `view` (color/size), `params` (type/default/position), `category` (determines `isJoin`, `isInput`, `isLoss`, `isSubFlow`).
- **NNTree** (conversion/nnTree.ts) — Converts visual graph to tree representation. Handles sequential chains, joins (multiple parents), loss nodes, and subflow containers with Kahn's topological sort. Subflows are type `"subflow"` with `entryNode` + internal `nodes` map (not flattened to sequential). Supports recursive nested subflows via `compileSubflowGraph`. Output JSON consumed by Python side.
- **TypeEngine** (conversion/typeEngine.ts) — Constraint-based static tensor type checker. Interprets `type_signature` from stereotype JSON. Data-driven — no hardcoded module-specific logic. Implements pattern matching with symbolic dimension binding, wildcard capture, param reference resolution, and dtype propagation. Supports: Input, Linear, ReLU (Phase 1). Join/subflow inference deferred to Phase 3/4.

  **Phase 2 (computed dims)**: The engine supports `computed` dimension patterns with formula resolution (`conv2d_hw`, `pool2d_hw`, `flatten_prod`). Conv2d output H/W are computed from kernel/stride/padding/dilation parameters. Flatten computes the product of wildcard-captured dimensions.

  **Phase 3 (join type checking)**: Join nodes with `kind: "join"` in their type_signature undergo multi-input pattern matching. Symbolic unification validates constraints (e.g., `MatMul`: K must match across inputs). The `Concat` constraint type sums dimensions on a specified axis. ScaledDotProduct validates Q/K/V shape compatibility.

  **Phase 5 (editor integration)**: The TypeEngine is wired into the visual editor. Edge connections, parameter changes, and diagram loads trigger real-time type inference. Errors appear as red badges on nodes and in a panel at the bottom of the Sidebar. Hovering an output handle shows the inferred output shape via tooltip.
- **Tensor Types** (conversion/tensortypes.ts) — Type model: ShapeDimension (const/symbolic/param_ref/wildcard discriminated union), TensorType (shape + dtype), TypeSignature (declarative input/output patterns), TypeEnvironment (symbolic bindings), TypeResult (annotations + errors).
- **FlowCanvas.svelte** — Main editor component. Renders SvelteFlow canvas, toolbar (Save/Load/Convert), and Sidebar. Three node types: `custom`, `subflow`, `join`.
- **Sidebar.svelte** — Node create/edit form. Resizable. Updates Diagram state reactively.

### Node Types

| Type | Svelte Component | Behavior |
|------|-----------------|----------|
| `custom` | CustomNode.svelte | Standard NN module (Linear, Conv2d, ReLU, etc.). Single input handle, single output handle. |
| `join` | JoinNode.svelte | Merge node (Addition, Einsum). Multiple input handles, single output. |
| `subflow` | SubflowNode.svelte | Container for nested nodes. Collapsible, hierarchical parent/child relationships. |

### Connection Rules

- Each target handle accepts only one connection (enforced by `checkValidConnection` in `utils.ts`)
- Source handles allow unlimited outgoing connections (implicit forks)
- Dragging nodes into subflow containers auto-reparents (handled by `onNodeDragStop`)
- Ancestry loop detection prevents circular parent references

## Stereotype System

### Categories

Each JSON in `Stereotypes/Modules/` has a `category` field that determines node behavior:

- `"Input"` — Single input node (auto-spawned). No input handle, green circle
- `"Linear"`, `"Conv2d"`, etc. — Standard modules. Input handle on top, output on bottom
- `"Join"` (in `Stereotypes/Joins/`) — Multi-input merge nodes
- `"SubFlow"` (in `Stereotypes/SubFlows/`) — Container templates with params like `iterations`/`n`
- Categories ending in `"Loss"` — Output nodes (no output handle, sets taskType for metric selection)

### Parameter Positions

Params in stereotype JSON can have `position: "top"` or `"bottom"` for display placement on the node. Default (no position) renders inline.

### Full Stereotype List

**Modules (27):** Input, Linear, Conv2d, ReLU, Tanh, Sigmoid, Softmax, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm, Flatten, MaxPool2d, AvgPool2d, Embedding, MultiheadAttention, Transformer, TransformerEncoderLayer, TransformerDecoderLayer, Unsample, Fork, PositionalEncoding, SequencePool, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

**Joins (6):** Addition, Einsum, MatMul, ScaledDotProduct, Concat, MaskedScaledDotProduct

**SubFlows (2):** Repeat (iterations param), HorizontalRepeat (n param)

**Ops (11):** Addition, Einsum, MatMul, ScaledDotProduct, Concat, Subflow, Repeat, HorizontalRepeat, MaskedScaledDotProduct, PositionalEncoding, SequencePool

### Type Signatures

Each stereotype JSON can optionally include a `type_signature` field declaring the module's tensor shape contract:

```json
{
  "type_signature": {
    "kind": "module",
    "input": [
      { "kind": "symbolic", "name": "$B" },
      { "kind": "wildcard" },
      { "kind": "param_ref", "name": "in_features" }
    ],
    "output": [
      { "kind": "symbolic", "name": "$B" },
      { "kind": "wildcard" },
      { "kind": "param_ref", "name": "out_features" }
    ]
  }
}
```

**Dimension kinds**: `const` (literal int), `symbolic` (e.g. `$B` for batch), `param_ref` (references node param), `wildcard` (matches zero or more arbitrary dims).

**Phase 1-3 modules** with type signatures: Input, Linear, ReLU, Tanh, Sigmoid, Softmax, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm, Conv2d, MaxPool2d, AvgPool2d, Flatten, Embedding (+6 joins: Addition, Concat, MatMul, ScaledDotProduct, MaskedScaledDotProduct). Modules without a `type_signature` emit a warning and propagate an unknown type — the system supports gradual typing.

The TypeEngine interprets these declarative signatures via constraint-based inference. Adding a new module requires only updating its stereotype JSON — no TypeScript changes needed.

## Design Requirements

From `analysis/requirements/reqs.md` — the DSL spec:

### Nodes

- **Module** — Receives 1 connection, can output to N nodes. Associated with expression language (code run when token passes through).
- **Input** — Only outgoing connection, no input.
- **Output/Loss** — Loss functions or undefined outputs. User chooses model output type.
- **SubModel** — Loadable from existing model file. Definable inside model. Saveable individually to disk. Nestable (submodels inside submodels).

### Stereotypes

- **Expression stereotype** — Defines module expression (not applicable to submodels).
- **Behavioral stereotype** — Extends module/submodel behavior. Examples: repeat module N times in sequence; fork input to N parallel copies then join.

### Joins

- **Forks** are implicit (any node can connect to N nodes).
- **Joins** must be explicit (module accepts 1 input). Join is a special Node with N inputs, 1 output.
- Join has an expression defining the operation (inner product, outer product, sum, any tensor op).

## Conversion Pipeline

```
NNTree JSON → convert.py → Hydra YAML configs (net/, optimizer/, trainer/, wandb/, dataset/, early_stopping/)
```

- **convert.py** — Reads NNTree JSON, generates Hydra-compatible config directory. Parses param strings via `ast.literal_eval`. Builds `_target_` paths for Hydra instantiation. Handles `type:"subflow"` nodes with `_recursive_: false` config. Supports `--num-classes` and `--dataset` CLI flags.
- **net/base.py** — `Net` class (LightningModule). Dynamically builds `ModuleDict` from config nodes. Topological sort for forward pass (BFS with in-degree tracking). Handles sequential chains, joins, subflows. Flatten is explicit via Flatten stereotype (no auto-flatten heuristic). Detects taskType (classification/regression) for metric selection.
  - **Join input ordering**: Join nodes receive inputs ordered by `targetHandle` ("in-0", "in-1", ...) preserved from diagram edges, not BFS traversal order. The `inputs` field lists parent node IDs in handle order. Non-commutative joins (MatMul, ScaledDotProduct) depend on this for correct behavior.
- **ops/addition.py**, **ops/einsum.py**, **ops/concat.py** — Custom join operations for forward pass.
- **dataset/** — MNIST, AutoencoderMNIST, and EnronSpam text classification dataset classes. Selectable via `--dataset` flag.
- **main.py** — Training entry point. Hydra-powered config with wandb logging, early stopping support.

## Project History

How this project evolved (from git log):

### Phase 1 — Foundation (commits b033849 → e9148f0)

Initial conversion from visual diagrams to Python code. First NNTree implementation. MNIST training pipeline working end-to-end. Early experiments with conversion logic.

### Phase 2 — Pipeline Maturation (commits b48dc0d → aaabec7)

Systematic hardening of the Python codegen pipeline:

- Fixed 12 issues in converted/ pipeline (a32611a)
- Replaced hardcoded stereotype names with pattern matching (52825cc)
- Replaced hardcoded join types with dynamic instantiation (fd32dde)
- Added taskType detection + `--num-classes` CLI (e99bc7d)
- Dynamic dataset selection via CLI + autoencoder dataset (b41d068)
- Configurable early stopping + max-epochs from CLI (aaabec7)
- Namespaced pythonClassName support (4d3ebd3)

### Phase 3 — SubFlow + Collapse (commits 50fe54a → db3b61b)

Added SubFlow visual container support:

- Collapse feature: toggle subflow visibility, hide/show children, auto-resize (50fe54a → db3b61b)
- Repeat SubFlow stereotype template (16d1304)
- Refactored subflow header: colored bar, removed floating badge (b33c1f7)

### Phase 4 — Autoencoder (commits 543f202, merged PR #5)

Added Unsample module stereotype (`nn.Upsample`) and full autoencoder submodel configuration. Merged autoencoder branch with diagram JSON examples for encoder-decoder architecture.

### Phase 5 — Test Suite & Code Hardening (commits 4fd2661 → 5414e3d)

Added vitest regression suite for the compiler core:

- Vitest v4 configured with Svelte Vite plugin
- 33 tests covering nnTree compilation (sequential, skip, autoencoder, error cases) and connection validation
- Replaced TestDiagram mock with real Diagram class (Svelte 5 `$state.raw` works via Vite plugin in vitest)
- Fixed 4 svelte-check type errors (addJoinNode config, unknown casts, empty fn stub)
- Node/edge factory helpers reducing fixture boilerplate ~60%

### Phase 6 — Subflow Compilation + Inference Tooling (commits 4026662 → 1c00a36)

- **Subflow compilation**: nnTree.ts extended with subflow boundary detection, Kahn's topological sort for internal graphs (compileSubflowLayers), Repeat stereotype unrolling (unrollLayers). 13 new tests covering boundary mapping, ×3 unrolling, subflow-in-chain, edge cases (empty/cycle/no-stereotype).
- **Dataset fix**: autoencoder_mnist.py no longer flattens images (removed view(-1)), returns native [C,H,W] — works for both Conv2d and Linear networks via generic flatten detection in base.py.
- **Inference script** (src/infer.py): --output saves predictions as JSON, --image-dir saves PNG montage + per-sample strips. Handles both classification and autoencoder tasks.
- **README rewrite**: converted/README.md replaced generic tutorial with actual usage docs (setup, convert.py args, main.py args, pipeline example, architecture).
- **Test count**: 33 → 48 tests.

### Phase 7 — Subflow Type Preservation + Python Ops (commits 271a796 → 7337303)

Subflows are now a distinct tree type instead of flattened sequential:

- **SubflowData**: new interface with `type: "subflow"`, `entryNode`, internal `nodes` map preserving topology
- **Nested subflows**: recursive `compileSubflowGraph` replaces flat `compileSubflowLayers`
- **Hidden nodes**: removed `!n.hidden` filter — collapsed subflow internals still compile
- **Python ops**: `ops.Subflow` (BFS internal graph execution), `ops.Repeat` (N copies via Sequential)
- **Hydra integration**: subflow nodes use `_recursive_: false` to prevent recursive instantiation
- **Test count**: 48 → 73 tests

### Phase 8 — Fork + Flatten + Concat + HorizontalRepeat (commit 548235c → 38169e9)

New stereotypes and refactoring:

- **Fork stereotype**: passthrough node for implicit forks, category "Fork", enables skip connections inside subflows (e.g. residual block with Repeat)
- **Flatten explicit**: removed auto-flatten heuristic from base.py, Flatten is now an explicit node (Stereotype already existed)
- **Concat join**: torch.cat(tensors, dim) for combining branches, enables building multihead attention from primitives
- **HorizontalRepeat stereotype**: `Stereotypes/SubFlows/HorizontalRepeat.json`, `ops.HorizontalRepeat`, N parallel subflow copies via `vmap` + `functional_call` + `stack_module_state`, output `[batch, ..., n*d]`
- **HorizontalRepeat join**: hardcoded to concat on dim=-1. Not configurable. See `horizontal_repeat.py:14` docstring.
- **Example diagram**: `horizontal_multihead_attention.json` — single head attention wrapped in HorizontalRepeat n=4
- **EnronSpamDataset**: text classification dataset via HF datasets + transformers
- **Test count**: 73 → 76 tests (3 Fork tests)

## Key Files Reference

| File | Purpose |
|------|---------|
| `front-end/src/Diagram.svelte.ts` | Reactive state: nodes, edges, add/delete/toggle |
| `front-end/src/stereotype.ts` | Load JSON stereotypes via glob |
| `front-end/src/conversion/nnTree.ts` | Graph → tree conversion |
| `front-end/src/conversion/tensortypes.ts` | Tensor type model: ShapeDimension, TensorType, TypeSignature, TypeResult |
| `front-end/src/conversion/typeEngine.ts` | Constraint-based type inference: pattern matching, symbolic binding, dtype propagation |
| `docs/designs/tensor-type-system/` | Full architectural design (7 docs): architecture, type model, engine spec, computed dims, join checking, editor integration, review |
| `front-end/src/FlowCanvas.svelte` | Main editor + toolbar |
| `front-end/src/Sidebar.svelte` | Node create/edit form |
| `front-end/src/nodes/SubflowNode.svelte` | Collapsible subflow UI |
| `front-end/src/utils.ts` | Connection validation |
| `front-end/src/__tests__/typeEngine.test.ts` | Type inference unit tests (18 tests, 5 skipped for Phase 2+) |
| `converted/src/convert.py` | NNTree JSON → Hydra configs |
| `converted/src/infer.py` | Inference: load trained model, run test set, save predictions/images |
| `converted/src/net/base.py` | Dynamic LightningModule |
| `converted/src/ops/addition.py` | Add join operation |
| `converted/src/ops/einsum.py` | Einsum join operation |
| `converted/src/ops/subflow.py` | Subflow base class: BFS internal graph execution |
| `converted/src/ops/repeat.py` | Repeat subgraph N times with independent weights |
| `converted/src/ops/horizontal_repeat.py` | HorizontalRepeat: N parallel copies via vmap, output `[batch, ..., n*d]`. **Join hardcoded to concat on dim=-1.** |
| `converted/src/ops/masked_scaled_dot_product.py` | Masked scaled dot product attention |
| `converted/src/ops/positional_encoding.py` | PositionalEncoding operation |
| `converted/src/ops/sequence_pool.py` | SequencePool (mean over seq dim) operation |
| `converted/src/dataset/enron_spam.py` | EnronSpam text classification dataset (HF datasets + transformers) |
| `horizontal_multihead_attention.json` | 4-head attention via HorizontalRepeat subflow + Concat |
| `transformer_classifier.json` | Full transformer: embed, posenc, encoder×2, pool, linear classifier |
| `front-end/vitest.integration.config.ts` | Integration test vitest config (tier filtering, Python pool) |
| `front-end/src/__tests__/integration/helpers.ts` | Shared integration helpers (manifest, uvRun, pipeline stages) |
| `front-end/src/__tests__/integration/smoke.test.ts` | Tier 0: nnTree compilation smoke tests |
| `front-end/src/__tests__/integration/convert.test.ts` | Tier 1: convert.py YAML generation tests |
| `front-end/src/__tests__/integration/forward.test.ts` | Tier 2: Net.forward() pass tests |
| `front-end/src/__tests__/integration/train.test.ts` | Tier 3: main.py training smoke tests (slow, CPU/GPU) |
| `front-end/src/__tests__/integration/infer.test.ts` | Tier 4: infer.py output validation tests |
| `converted/src/tests/test_main.py` | Python training smoke tests (autoencoder + MNIST) |
| `converted/src/tests/test_infer.py` | Python inference validation tests |
| `converted/src/tests/test_integration.py` | Python integration tests (convert → Net forward end-to-end) |
| `examples/manifest.json` | Manifest of example diagrams/nntrees (input shapes, task types, trainable flags) |
