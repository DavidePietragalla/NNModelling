# NNModelling

DSL for designing neural networks via visual node editor. Diagrams convert to PyTorch/Lightning training pipelines.

```
Stereotypes/ (JSON) → Svelte Flow Editor → NNTree (JSON) → convert.py → Hydra YAML → main.py → Training
                                                                                       → infer.py  → Inference
```

## Repository Structure

```
NNModelling/
├── front-end/          # Svelte 5 + Svelte Flow visual editor (TypeScript)
├── converted/          # Python codegen target (PyTorch + Lightning + Hydra)
├── mcp-server/         # MCP server — thin proxy for browser diagram state via WebSocket RPC
├── Stereotypes/        # JSON template definitions (Modules, Joins, SubFlows)
├── docs2/              # Sphinx documentation (user guide, architecture, API reference)
├── examples/           # Test fixtures and diagrams for integration tests
├── analysis/           # UML + requirements documentation
├── *.json              # Example diagrams (Svelte Flow format)
├── CLAUDE.md           # Detailed project guide for AI agents
├── AGENTS.md           # Project guide (alias for CLAUDE.md)
└── package.json        # pnpm workspace root
```

## Quick Start

### Frontend (Editor)

```bash
cd front-end
npm install
npm run dev
```

### Backend (Training)

```bash
cd converted
uv sync
uv run python src/convert.py <nn_tree_json> <output_dir>
uv run python src/main.py --config-dir <dir>
```

### MCP Server

```bash
cd mcp-server
pnpm install && pnpm run build
pnpm run start
```

## Key Concepts

- **Nodes**: Layers (Linear, Conv2d, ReLU...), Joins (Addition, Concat, MatMul...), SubFlows (Repeat, HorizontalRepeat), Loss (CrossEntropyLoss...)
- **Edges**: Data flow between nodes. Forks implicit, joins explicit.
- **NNTree**: Intermediate representation — compiled DAG preserving sequential chains, join ordering, and recursive subflows.
- **SubFlows**: Containers with internal graph topology. Repeat (sequential N times with independent weights) and HorizontalRepeat (parallel N copies via vmap).
- **Join ordering**: Non-commutative joins (MatMul, ScaledDotProduct) receive inputs ordered by edge targetHandle, not BFS arrival.
- **Stereotypes**: JSON files defining node category, Python class mapping, view defaults, and configurable parameters.
- **MCP Server**: Thin proxy that enables LLM agents to manipulate the diagram via WebSocket RPC to the browser.

## Documentation

- `CLAUDE.md` / `AGENTS.md` — Full project guide for AI agents (architecture, test patterns, stereotype list)
- `docs2/build/html/index.html` — Sphinx documentation (open in browser): user guide, architecture, API reference, examples

## Testing

```bash
# Frontend unit tests
cd front-end && npm run test

# Integration tests (tiered: compile → convert → forward → train → infer)
cd front-end && npm run test:integration

# Python tests
cd converted && uv run pytest src/tests/ -v
```
