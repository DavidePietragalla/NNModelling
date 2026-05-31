# NNModelling

DSL for designing neural networks via visual node editor. Diagrams convert to PyTorch/Lightning training pipelines.

```
Stereotypes/ (JSON) → Svelte Flow Editor → NNTree (JSON) → convert.py → Hydra YAML → main.py → Training
```

## Repository Structure

```
NNModelling/
├── front-end/          # Svelte 5 + Svelte Flow visual editor (TypeScript)
├── converted/          # Python codegen target (PyTorch + Lightning + Hydra)
├── Stereotypes/        # JSON template definitions (Modules, Joins, SubFlows)
├── *.json              # Example diagrams (Svelte Flow format)
├── analysis/           # UML + requirements documentation
└── CLAUDE.md           # Detailed project guide
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

## Key Concepts

- **Nodes**: Modules (Linear, Conv2d, ReLU...), Joins (Addition, Concat, MatMul...), SubFlows (Repeat, HorizontalRepeat)
- **Edges**: Data flow between nodes. Forks implicit, joins explicit.
- **NNTree**: Intermediate representation — compiled DAG preserving sequential chains, join ordering, and recursive subflows.
- **SubFlows**: Containers with internal graph topology. Repeat (sequential N times) and HorizontalRepeat (parallel N copies via vmap).
- **Join ordering**: Non-commutative joins (MatMul, ScaledDotProduct) receive inputs ordered by edge targetHandle, not BFS arrival.

## Documentation

See `CLAUDE.md` for full architecture, test patterns, and stereotype list.
