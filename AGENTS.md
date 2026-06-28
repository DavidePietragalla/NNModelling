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
npm run test      # Run vitest (single run)
npm run test:watch  # Run vitest (watch mode)
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
│   │   └── utils.test.ts       # checkValidConnection tests
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
│   └── infer.py                # Inference on trained model (--output, --image-dir)
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

### Testing

- **Framework**: Vitest (v4), configured in `vitest.config.ts`
- **Pattern**: Pure TS unit tests, no DOM/browser
- **Real Diagram**: Tests use real `Diagram` class (Svelte `$state.raw` compiled by Vite plugin). Stub `globalThis.window` before construction.
- **Helpers**: `node(id, stereo, name, params, overrides?)` and `edge(id, source, target, handles?)` for concise fixtures.
- **Commands**: `pnpm test` (run), `pnpm test:watch` (watch)
- **Coverage**: 76 tests — sequential chain, skip/joins, autoencoder, subflow compilation, nested subflows, hidden nodes, error handling, connection validation, Fork node, Python ops (36), convert.py (35), net/base.py (21), integration

### Front-end Data Flow

```
Stereotypes/ (JSON) → Stereotype class → Diagram class (reactive state) → FlowCanvas.svelte (Svelte Flow UI)
                                                                               ↓
                                                                          NNTree class (conversion)
                                                                               ↓
                                                                          JSON → convert.py (Hydra configs)
```

### Key Classes

- **Diagram.svelte.ts** — Central state manager. Holds `nodes` and `edges` as Svelte 5 `$state.raw` arrays. Methods: `addModule`, `addJoinNode`, `addSubGraph`, `deleteNodes`, `importFromJson`, `exportToJson`, `toggleSubflow`.
- **Stereotype** (stereotype.ts) — Loads all JSON files from `Stereotypes/**/*.json` via Vite's `import.meta.glob`. Each stereotype defines: `pythonClassName`, `view` (color/size), `params` (type/default/position), `category` (determines `isJoin`, `isInput`, `isLoss`, `isSubFlow`).
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
