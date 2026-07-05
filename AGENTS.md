# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**NNModelling** — DSL for designing neural networks via visual node editor. Diagrams convert to PyTorch/Lightning code.

Three main packages (pnpm workspace):

1. **front-end/** — Svelte 5 + Svelte Flow visual editor (TypeScript)
2. **converted/** — Python codegen target (PyTorch + Lightning + Hydra)
3. **mcp-server/** — MCP server — thin proxy that queries browser diagram state via WebSocket RPC

### Tech Stack

- **Frontend**: Svelte 5, Svelte Flow (@xyflow/svelte), Vite 8, TypeScript
- **Build**: pnpm v10 (monorepo workspace)
- **MCP Server**: @modelcontextprotocol/sdk, ws, zod
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

# Documentation (Sphinx + TypeDoc)
cd docs2 && uv run make html  # Build Sphinx HTML docs
# Or from root:
pnpm run docs               # Full build: TypeDoc + Sphinx
bash gendocs.sh             # Sphinx-only build (auto-creates uv venv)
```

```bash
cd converted
uv run python src/convert.py <nn_tree_json> <output_dir>  # Generate Hydra configs
uv run python src/main.py --config-dir <dir>               # Train model
uv run python src/infer.py --config-path <dir> --config-name <name> --weights <path>  # Inference
```

```bash
cd mcp-server
pnpm run build       # Compile TypeScript
pnpm run test        # Run unit tests (vitest)
pnpm run start       # Start server (node dist/index.js)

# Run MCP server directly with tsx (for development, bypasses ESM issues):
npx tsx mcp-server/src/index.ts
```

## Project Structure

```
NNModelling/
├── front-end/src/              # Svelte 5 visual editor
│   ├── __tests__/              # Vitest test suites
│   │   ├── helpers.ts          # Test factories (stubWindow, node, edge)
│   │   ├── nnTree.test.ts      # NNTree regression tests
│   │   ├── utils.test.ts       # checkValidConnection tests
│   │   ├── BrowserRPCHandler.test.ts  # RPC handler tests
│   │   ├── undoRedo.test.ts     # Undo/redo snapshot-based tests
│   │   └── integration/        # Integration test suite (tiered, Python pipeline)
│   │       ├── helpers.ts      # Shared helpers (manifest, uvRun, pipeline stages)
│   │       ├── smoke.test.ts   # Tier 0: nnTree compilation
│   │       ├── convert.test.ts # Tier 1: convert.py YAML generation
│   │       ├── forward.test.ts # Tier 2: Net.forward() pass
│   │       ├── train.test.ts   # Tier 3: main.py training smoke
│   │       └── infer.test.ts   # Tier 4: infer.py output validation
│   ├── core/                     # Pure TypeScript (no Svelte deps)
│   │   ├── DiagramCore.ts        # All diagram business logic + EventBus
│   │   ├── EventBus.ts           # Typed event emitter with monotonic seq
│   │   ├── StereotypeCore.ts     # Pure TS stereotype with dual loader (Vite/Node)
│   │   ├── types.ts              # Shared type definitions (events, WS, configs)
│   │   └── validation.ts         # Standalone connection validation
│   ├── sync/                     # Browser-side RPC handler
│   │   └── BrowserRPCHandler.ts  # Responds to MCP server RPC calls
│   ├── nodes/
│   │   ├── CustomNode.svelte   # Standard NN module node
│   │   ├── JoinNode.svelte     # Merge node (multi-input)
│   │   └── SubflowNode.svelte  # Collapsible submodel container
│   ├── conversion/
│   │   └── nnTree.ts           # Diagram → tree representation
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
├── mcp-server/                        # MCP server package (Node.js ESM)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vitest.config.ts
│   ├── typedoc.json                 # TypeDoc configuration
│   └── src/
│       ├── index.ts                   # Entry point (stdio transport)
│       ├── server.ts                  # MCP server setup + tool registration
│       ├── browser-client.ts          # WebSocket RPC client (multi-tab)
│       ├── errors.ts                  # Error classes (5: base + 4 pipeline)
│       ├── pipeline.ts                # Python subprocess interface
│       └── tools/
│           ├── graph.ts               # create_node, delete_nodes, etc.
│           ├── parameters.ts          # set_parameter, update_parameters, etc.
│           ├── selection.ts           # select_nodes, clear_selection, etc.
│           ├── canvas.ts              # get_canvas_state, fit_view, center_view
│           ├── validation.ts          # validate_graph, validate_connections, etc.
│           ├── conversion.ts          # compile_nntree, execute_conversion, etc.
│           ├── inspection.ts          # get_graph, get_node, statistics, etc.
│           ├── connection.ts          # list_browser_tabs, select_browser_tab
│           └── lifecycle.ts           # reset_diagram, ping
├── docs/
│   ├── report/                 # Bug reports (mcp_issues.md, etc.)
│   └── designs/                # Design plans (mcp-sync-fixes/plan.md, etc.)
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
├── docs2/                       # Sphinx documentation infrastructure
│   ├── Makefile                 # Sphinx build targets (html, clean)
│   ├── requirements.txt        # Sphinx + theme dependencies
│   ├── source/
│   │   ├── conf.py             # Sphinx configuration (autodoc, napoleon, rtd theme)
│   │   ├── index.rst           # Root toctree
│   │   ├── user_guide.rst      # Educational user guide
│   │   ├── architecture.rst    # System architecture explanation
│   │   ├── stereotypes.rst     # All 35 stereotypes with params and examples
│   │   ├── python_api.rst      # Auto-generated Python API docs
│   │   ├── typescript_api.rst  # TypeDoc integration page
│   │   ├── examples.rst        # Walkthrough of all 10 example diagrams
│   │   └── _static/            # Static assets
│   └── build/                  # Build output (generated)
├── package.json                # Root workspace scripts (docs, etc.)
├── gendocs.sh                  # Documentation build script (auto-creates venv)
```

## Architecture

### Testing — Front-end (Vitest Unit Tests)

- **Framework**: Vitest (v4), configured in `vitest.config.ts`
- **Pattern**: Pure TS unit tests, no DOM/browser
- **Real Diagram**: Tests use real `Diagram` class (Svelte `$state.raw` compiled by Vite plugin). Stub `globalThis.window` before construction.
- **Helpers**: `node(id, stereo, name, params, overrides?)` and `edge(id, source, target, handles?)` for concise fixtures.
- **Coverage**: **120 tests** — sequential chain, skip/joins, autoencoder, subflow compilation, nested subflows, hidden nodes, error handling, connection validation, Fork node, BrowserRPCHandler, undo/redo, param merging, edge handle defaults, viewport injection

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
Stereotypes/ (JSON) → StereotypeCore → DiagramCore (pure TS state + EventBus)
                                            ↑ extends
                                       Diagram.svelte.ts ($state.raw wrapper)
                                            ↓
                                       FlowCanvas.svelte (Svelte Flow UI)
                                            ↓
                                       NNTree class (conversion)
                                            ↓
                                        JSON → convert.py (Hydra configs)
```

### MCP Server & Browser RPC

The MCP server is a thin proxy — it does NOT hold its own DiagramCore. All diagram state lives exclusively in the browser. The server sends RPC requests to the browser via WebSocket and the browser's `BrowserRPCHandler` executes them on its `DiagramCore`.

```
Browser (DiagramCore + $state.raw)  ←── single source of truth
    │
    │  WebSocket (ws://localhost:9339)
    │  BrowserRPCHandler receives: {id, method, params}
    │  BrowserRPCHandler responds:  {id, result} or {id, error}
    │
    ▼
MCP Server (thin proxy, no DiagramCore)
    │
    ├──▶ stdio (MCP protocol) ──▶ LLM Agent (manipulation via ~38 tools)
    │
    └──▶ Subprocess ──▶ Python Pipeline (convert.py, main.py, infer.py)
         Server queries browser for NNTree JSON → writes temp file → runs `uv run python`
```

**Multi-tab support**: Multiple browser tabs can connect simultaneously. Each tab gets a sequential ID ("tab_1", "tab_2", ...). The first tab is auto-selected. Use `list_browser_tabs` to see all tabs and `select_browser_tab` to switch.

**What was removed** (vs Phase 10):
- Server-side `DiagramCore` — no state duplication
- `EventBus` — no domain events needed on server
- `TransactionManager`, `HistoryManager` — unused
- `ws-server.ts` delta broadcast — replaced by simple RPC request/response
- All 14 MCP resources — replaced by tools (`get_graph`, `get_node`, etc.)
- 22 error classes → 5 (pipeline errors only)
- `analysis.ts` — graph statistics computed in browser

### Key Classes

- **DiagramCore** (core/DiagramCore.ts) — Pure TypeScript state authority. Holds `nodes` and `edges` as plain arrays. All business logic: `addModule`, `addJoinNode`, `addSubGraph`, `deleteNodes`, `addEdge`, `moveNode`, `importFromJson`, `exportToJson`, `toggleSubflow`, `undo`, `redo`. Integrates `EventBus` — every mutation emits typed domain events. Snapshot-based undo/redo (Ctrl+Z/Ctrl+Alt+Z) via `getSnapshot`/`restoreSnapshot` with 50-entry stack limit.
- **Diagram.svelte.ts** — Thin Svelte 5 wrapper extending `DiagramCore`. Overrides `nodes`/`edges` with `$state.raw` for reactive UI. Handles Svelte-specific concern: auto-spawn Input node. (No callback re-hydration needed — SubflowNode uses `getContext`.)
- **StereotypeCore** (core/StereotypeCore.ts) — Pure TypeScript stereotype with dual loader: `loadFromDirectory()` uses Vite's `import.meta.glob` for browser; `loadFromDirectoryNode(path)` uses `fs.readdirSync` for Node.js/MCP server.
- **Stereotype** (stereotype.ts) — Thin wrapper extending `StereotypeCore`. Delegates to Vite loader via `StereotypeCore.loadFromDirectory()`.
- **NNTree** (conversion/nnTree.ts) — Converts visual graph to tree representation. Handles sequential chains, joins (multiple parents), loss nodes, and subflow containers with Kahn's topological sort. Subflows are type `"subflow"` with `entryNode` + internal `nodes` map (not flattened to sequential). Supports recursive nested subflows via `compileSubflowGraph`. Output JSON consumed by Python side.
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

The `category` field in every stereotype JSON determines the node's role and handle configuration:

| Category | Role | Handles |
|----------|------|---------|
| `"Input"` | Network entry point (auto-spawned) | 0 in, 1 out |
| `"Fork"` | Passthrough for explicit branching inside subflows | 1 in, 1 out |
| `"Layer"` | Standard module (Linear, Conv2d, ReLU, Dropout, ...) | 1 in, 1 out |
| `"Loss"` | Loss function / output node (BCELoss, CrossEntropyLoss, ...) | 1 in, 0 out |
| `"Join"` | Multi-input merge node (Addition, Concat, MatMul, ...) | N in, 1 out |
| `"Subflow"` | Container holding a sub-graph with structural transformation | 1 in, 1 out |
| `"Module"` | Generic; reserved for future use | Depends |

**Note**: `"expr"` top-level field was removed from all stereotypes (Einsum's `params.expr` is preserved as a user parameter). Categories like `"Activation"`, `"Normalization"`, `"Pooling"`, etc. were consolidated into `"Layer"`.

The StereotypeCore class provides two loaders: `loadFromDirectory()` (Vite glob, for browser) and `loadFromDirectoryNode(path)` (Node fs, for MCP server).

### Parameter Positions

Params in stereotype JSON can have `position: "top"` or `"bottom"` for display placement on the node. Default (no position) renders inline.

### Full Stereotype List

**Modules (27):** Input, Linear, Conv2d, ReLU, Tanh, Sigmoid, Softmax, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm, Flatten, MaxPool2d, AvgPool2d, Embedding, MultiheadAttention, Transformer, TransformerEncoderLayer, TransformerDecoderLayer, Unsample, Fork, PositionalEncoding, SequencePool, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

**Joins (6):** Addition, Einsum, MatMul, ScaledDotProduct, Concat, MaskedScaledDotProduct

**SubFlows (2):** Repeat (iterations param), HorizontalRepeat (n param)

**Ops (11):** Addition, Einsum, MatMul, ScaledDotProduct, Concat, Subflow, Repeat, HorizontalRepeat, MaskedScaledDotProduct, PositionalEncoding, SequencePool

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

### Phase 9 — Core Extraction for MCP Server (commit XXXXXXX)

Refactored frontend to extract pure TypeScript core — preparation for the MCP server (Phase 2):

- **core/DiagramCore.ts**: All business logic from Diagram.svelte.ts with zero Svelte deps. Integrates EventBus — every mutation (addModule, deleteNodes, addEdge, etc.) emits typed domain events. New methods: addEdge, removeEdge, reconnectEdge, moveNode, moveNodes, getSnapshot, restoreSnapshot, selectNodes, clearSelection.
- **core/EventBus.ts**: Typed event emitter with monotonic sequence numbers, ring buffer (max 1000 events), on/onAny/emit/getEventsSince.
- **core/StereotypeCore.ts**: Pure TS stereotype with dual loader: Vite's import.meta.glob for browser, fs.readdirSync for Node.js/MCP server.
- **core/types.ts**: Shared type definitions: DomainEvent<T> with 12 event types, WebSocket delta protocol types (WSSnapshotMessage, WSDeltaMessage, DeltaOperation), config interfaces.
- **core/validation.ts**: Standalone checkValidConnection on plain Edge[] (was coupled to Diagram instance).
- **Diagram.svelte.ts**: Reduced from 346 to 22 lines (latest: callback elimination). Now a thin Svelte wrapper extending DiagramCore — only adds $state.raw reactivity and auto-spawn Input.
- **stereotype.ts**: Reduced from 93 to 17 lines. Stereotype extends StereotypeCore.
- **nnTree.ts**: Now accepts DiagramCore (type-only import) instead of Diagram.
- **utils.ts**: Delegates checkValidConnection to core/validation.ts.
- **Verification**: All 76 unit tests pass, 64 integration smoke tests pass, svelte-check unchanged (4 pre-existing errors, 7 warnings).
- **Git tag**: `phase1-complete`

### Phase 10 — MCP Server & Real-Time Sync

Built the initial MCP server with DiagramCore on server + delta sync. Included: 43 tools, 14 resources, EventBus, TransactionManager, HistoryManager, ws-server delta broadcast, DiagramSyncClient.

### Phase 11 — MCP Server Simplification (phase1-complete)

Removed server-side state duplication. The server is now a thin proxy:
- Deleted server-side DiagramCore, EventBus, TransactionManager, HistoryManager
- Deleted delta broadcast (ws-server.ts), replaced with simple WebSocket RPC
- Deleted DiagramSyncClient, replaced with BrowserRPCHandler
- Deleted all 14 MCP resources (replaced by tools)
- Deleted 3 tool files (transaction, history, events)
- Reduced error classes from 22 to 5
- Added multi-tab support: list_browser_tabs, select_browser_tab
- Reduced mcp-server from ~4700 to ~1100 lines (~3600 removed)

| File | Purpose |
|------|---------|
| `front-end/src/Diagram.svelte.ts` | Thin Svelte wrapper: $state.raw, auto-spawn Input |
| `front-end/src/stereotype.ts` | Stereotype extends StereotypeCore, Vite loader |
| `front-end/src/core/DiagramCore.ts` | Pure TS state management + EventBus integration |
| `front-end/src/core/EventBus.ts` | Typed event emitter with monotonic sequencing |
| `front-end/src/core/StereotypeCore.ts` | Pure TS stereotype with dual Vite/Node.js loader |
| `front-end/src/core/types.ts` | Shared type definitions (DomainEvent, WS messages, configs) |
| `front-end/src/core/validation.ts` | Standalone connection validation on Edge[] |
| `front-end/src/conversion/nnTree.ts` | Graph → tree conversion |
| `front-end/src/sync/BrowserRPCHandler.ts` | Browser-side RPC handler |
| `front-end/src/FlowCanvas.svelte` | Main editor + toolbar |
| `front-end/src/Sidebar.svelte` | Node create/edit form |
| `front-end/src/nodes/SubflowNode.svelte` | Collapsible subflow UI |
| `front-end/src/utils.ts` | Connection validation |
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
| `mcp-server/src/server.ts` | MCP server bootstrap + tool registration |
| `mcp-server/src/browser-client.ts` | WebSocket RPC client (multi-tab support) |
| `mcp-server/src/index.ts` | Entry point (stdio transport) |
| `mcp-server/src/tools/graph.ts` | Graph manipulation tools |
| `mcp-server/src/tools/parameters.ts` | Parameter management tools |
| `mcp-server/src/tools/selection.ts` | Node selection tools |
| `mcp-server/src/tools/canvas.ts` | Canvas state + viewport tools |
| `mcp-server/src/tools/validation.ts` | Graph validation tools |
| `mcp-server/src/tools/conversion.ts` | Conversion pipeline tools |
| `mcp-server/src/tools/inspection.ts` | Inspection tools |
| `mcp-server/src/tools/connection.ts` | Tab management tools |
| `mcp-server/src/tools/lifecycle.ts` | Diagram lifecycle tools |
| `mcp-server/src/errors.ts` | Error classes (5: base + 4 pipeline) |
| `mcp-server/src/pipeline.ts` | Python subprocess interface |
| `pnpm-workspace.yaml` | pnpm monorepo config |

### Phase 12 — Undo/Redo System (current)

Snapshot-based undo/redo for the visual editor:

- **Undo/Redo engine**: Added to DiagramCore — `_undoStack`, `_redoStack`, `_captureEnabled` flag, `undo()`, `redo()` methods. Every mutation method captures state via `_captureUndoState()` before modifying.
- **Snapshot-based**: Uses existing `getSnapshot()`/`restoreSnapshot()` — no per-event revert functions needed.
- **Keyboard shortcuts**: Ctrl+Z (undo) and Ctrl+Alt+Z (redo) in FlowCanvas.svelte, with input field guard.
- **Stack limit**: 50 entries max; `_redoStack` cleared on new mutations.
- **Bug fix**: `toggleSubflow` recursive calls no longer duplicate undo captures (extracted `_toggleSubflowRecursive` helper).
- **Bug fix**: `addEdge` captures only after validation pass (no undo slot wasted on rejected connections).
- **Initial state**: Auto-spawned Input node is excluded from undo history.
- **Test count**: 86 → 96 tests (10 new undo/redo tests)

### Phase 13 — Category Cleanup + Documentation Rewrite

Fixed the stereotype category system and rewrote the documentation:

- **Fork.json**: category changed from `"Layer"` to `"Fork"` (no code changes needed — Fork is identified by stereotype name, not category)
- **`"expr"` field**: removed unused top-level `"expr": ""` from all 25 stereotype JSONs (Einsum's `params.expr` preserved as a user parameter)
- **`stereotypes.rst`**: full rewrite from 696 lines of repetitive per-stereotype listings to a 210-line educational Sphinx reference doc. New format:
  - Categories table with handle configuration
  - JSON Field Reference table (all 11 fields: type, required, default, description)
  - 3 exemplary stereotypes (Linear, Concat, Repeat) instead of all 35
  - Notes section with gotchas (loss nodes, flatten, join ordering, etc.)
- **No code changes needed**: all category-driven logic uses stereotype name or dedicated flags (`isLoss`, `isInput`, etc.), not raw category strings

### Phase 14 — MCP Sync Fixes (current)

Three MCP synchronization bugs fixed, discovered during agent-driven diagram creation:

| Bug | Severity | Root Cause | Fix |
|-----|----------|------------|-----|
| Edges invisible on canvas | High | Handle ID mismatch — `CustomNode.svelte` Handles had no `id`, SvelteFlow defaults to `null`; `DiagramCore.addEdge` defaults to `"out"`/`"in"` | Added `id="in"` / `id="out"` to CustomNode.svelte Handles |
| `fit_view`/`center_view` no-ops | Medium | `BrowserRPCHandler` stubs returned `{ success: true }` without calling SvelteFlow API | Injected `ViewportController` (fitView/setCenter) via constructor from FlowCanvas.svelte |
| `create_node` params lost | High | `addModule`/`addJoinNode` used all-or-nothing param assignment, discarding stereotype defaults | Added `_mergeNodeParams()` helper — starts from `getDefaultParams()` and overlays user values preserving `position` metadata |

**Key changes by file:**

- **`CustomNode.svelte`**: Added `id="in"` on target Handle, `id="out"` on source Handle. Removed duplicate `<Handle type="source">`.
- **`DiagramCore.ts`**: Added `_mergeNodeParams(stereotype, userParams?)` — deep merge: stereotype defaults → user overlay. Used in both `addModule` and `addJoinNode`.
- **`BrowserRPCHandler.ts`**: New `ViewportController` interface with `fitView()` and `setCenter()`. Constructor accepts optional third parameter. `handleFitView`/`handleCenterView` now call real SvelteFlow methods when available, with graceful no-op fallback.
- **`Diagram.svelte.ts`**: Added `graph_changed` event listener that forces Svelte 5 reactivity by spreading `this.nodes`/`this.edges` into new arrays — necessary because RPC mutates plain arrays, which `$state.raw` doesn't detect.
- **`FlowCanvas.svelte`**: Extracts `fitView`/`setCenter` from `useSvelteFlow()` and passes to `BrowserRPCHandler`. Added `$effect` calling `fitView()` on every `graph_changed` event for auto-centering after RPC mutations.

**Test count**: 96 → 120 (+24 tests for param merging, edge handle defaults, viewport injection)
