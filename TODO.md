# Todo List for NNModelling Extension & Hardening (Final Sprint)

## Overview & Architecture Context

Today's sprint focuses on two core engineering objectives:

1. **Defensive Testing & Code Hardening**: Setting up a test suite on the frontend before expanding the compiler core to prevent regression.
2. **SubFlow Compile Feature Completion**: Extending `nnTree.ts` and `convert.py` to handle nested recursive SubFlow containers driven by behavioral stereotypes (e.g., `Repeat` with iterations parameter).
3. **Architectural Guardrails**: Formally documenting and tackling the explicit vs. implicit abstraction discrepancy discovered during the architectural audit (the `Flatten` stereotype vs. backend auto-flattening paradox).

---

## 📅 Chronological Task List

### Phase 1: Test Setup & Base Hardening (DONE)

- [x] **Configure Vitest**: Install and configure Vitest inside the `front-end/` directory as the baseline unit testing framework.
- [x] **Establish Ground-Truth Snapshot**: Export a known-working diagram layout (e.g., `mninst.json` or `mnist_skips.json`) to serve as an immutable regression test asset.
- [x] **Write Core Compiler Test**: Feed the ground-truth JSON structure directly into `nnTree.ts` and assert that the generated abstract tree structure matches structural expectations identically.
- [x] **Validate Loop Detection**: Add an isolated unit test specifically checking that `checkValidConnection` inside `utils.ts` cleanly intercepts and throws errors on circular parent-child ancestry definitions.
- [x] **Fix svelte-check type errors**: 4 type errors fixed (addJoinNode config, unknown casts, empty fn stub).
- [x] **Refactor tests to use real Diagram**: Remove TestDiagram mock, use real Diagram with `$state.raw` via Svelte plugin in vitest.

### Phase 1b: Test Coverage Gaps (NEW)

- [x] **Self-loop edge case**: Add test for `checkValidConnection` with connection from node to itself. Currently not prevented — decide if should be.
- [x] **SubFlow compilation test**: Add NNTree test for `auto_encoder_submodels.json` — requires SubFlow support in nnTree.ts first.
- [ ] **convert.py integration test**: Feed NNTree JSON output into `convert.py` and verify generated Hydra YAML structure.

### Phase 2: SubFlow Stereotype Conversion (Est: 2-3 hrs) — DONE (nnTree.ts)

- [x] **Write Failing SubFlow Test Spec (TDD Approach)**: 13 tests across 4 describe blocks (boundary, Repeat ×3, chain, edge cases).
- [x] **Extend Frontend Compiler (`nnTree.ts`)**: 5 new methods (isSubflowNode, nodeToModule, compileSubflowLayers, unrollLayers, processSubflow). createSequential breaks at subflow boundaries.
- [ ] **Extend Backend Code Generator (`convert.py`)**: Still pending — convert.py doesn't handle subflow-specific stereotyping.
- [x] **Iterate Until Green**: 48/48 tests pass, 0 svelte-check errors.

### Phase 2b: Subflow Fixes — Nested Containers & Stereotype Metadata (NEW)

- [ ] **Nested subflow test**: Add test and fix `compileSubflowLayers` to recursively handle internal nodes that are themselves subflow containers (parentId of a different subflow). Currently internal subflow children are lost.
- [ ] **Stereotype metadata in output**: Preserve `stereotype` and `iterations` (or other params) in the NNTreeNode data for subflow nodes, so the JSON output retains behavioral stereotype info instead of silently unrolling into plain SequentialData.
- [ ] **Test subflow without unrolling**: Verify that a subflow with no stereotype (plain container, like `auto_encoder_submodels.json`) still skips Iterations entirely — i.e. `compileSubflowLayers` produces layers but no unrolling logic fires.

### Phase 2c: Sequence/Attention Data Support (NEW — blocker per Transformer)

- [ ] **Fix auto-flatten in `base.py:forward`**: L'auto-flatten cerca `"Linear"` in `_target_` e appiattisce qualsiasi input 3D+ a 2D prima del Linear. Per sequence data `(batch, seq, features)` questo distrugge la dimensione temporale. Serve una logica che non appiattisca quando il tensore è già 2D (batch, features) o quando il task è sequence-based.
- [ ] **Test 3D forward pass**: Creare un test che passa dati 3D `(2, 8, 64)` attraverso una catena Linear → Attention → Linear e verifica che le dimensioni `(batch, seq, d_model)` siano preservate.

### Phase 3: E2E Integration & Verification (Est: 1 hr)

- [ ] **Generate Hydra Test Configs**: Invoke `python src/convert.py <subflow_test_json> <output_dir>` and structurally verify that the dynamic YAML dictionary matches Hydra parsing specifications.
- [ ] **Execute Dynamic Forward Pass**: Run `python src/main.py --config-dir <dir>` using the generated SubFlow layouts to verify that the `BFS` topological sorting block in `net/base.py` sequences the layers flawlessly without execution or dimension deadlocks.

### Phase 4: Oral Exam Defense Preparation (Est: 30 mins)

- [ ] **Review `net/base.py` Code Base**: Thoroughly examine line-by-line how the Python runtime resolves topological ranks via in-degree tracking during forward sweeps.
- [ ] **Sanitize the Explicit vs. Implicit Flaw**: Reframe the implicit backend auto-flattening mechanism as a deliberate architectural feature. Prepare an explanation centered around *fault-tolerant compiler design* and *UX defensive programming* rather than missing specifications.

---

## 🤖 Optimized DeepSeek Prompts for Sprints

Use these highly specific prompts sequentially throughout the day to guide DeepSeek through code-generation tasks.

### 📋 Prompt 1: Vitest Configuration & Regression Tests Creation

**Context**: Run inside `front-end/src/` to lock down base compiler stability.

> Act as a Principal Software Engineer specializing in Compiler design and TypeScript testing.
> I need to add a Vitest suite to my Svelte 5 DSL web editor app.
>
> Here is the exact implementation of my frontend graph-to-tree compiler core (`front-end/src/conversion/nnTree.ts`):
> [PASTE YOUR nnTree.ts CONTENT HERE]
>
> And here is my graph connection utility module (`front-end/src/utils.ts`):
> [PASTE YOUR utils.ts CONTENT HERE]
>
> Tasks:
>
> 1. Generate a robust setup for Vitest targeting pure functions.
> 2. Write a comprehensive regression test suite that takes a standard mock Graph JSON object (representing nodes like 'Input', 'Linear', 'ReLU', 'Loss') and validates that `nnTree` correctly compiles it to an AST representation.
> 3. Write isolated unit tests for `checkValidConnection` ensuring it detects and throws exceptions on cyclic connections and ancestry reference loops.
> Do not hallucinate functions. Output strict, clean TypeScript code matching our data contracts.

### 🧩 Prompt 2: SubFlow Stereotype Compilation Logic Development

**Context**: Use this once base tests pass, to inject the new SubFlow expansion algorithm.

> Act as an Expert Compiler Engineer. We are extending our visual Deep Learning DSL to process 'subflow' nodes. A subflow node acts as a structural container that can group standard layer nodes and has a behavioral stereotype like 'Repeat' with an 'iterations' parameter (e.g., repeating the nested sub-graph sequence N times).
>
> Here is my current `nnTree.ts` compiler file that handles standard modules and explicit Join nodes:
> [PASTE YOUR CURRENT nnTree.ts FILE]
>
> Tasks:
>
> 1. Modify the compilation algorithm to recursively handle nodes inside subflows. If a subflow has a 'Repeat' stereotype, the child sub-graph must be unrolled or compiled sequentially based on the 'iterations' param.
> 2. Ensure that edges entering or leaving the Subflow bounds map properly to the first internal layer and last internal layer respectively.
> 3. Provide the accompanying Vitest spec to assert that this subflow flattening and unrolling logic executes safely without breaking baseline sequential node translations.

### 🐍 Prompt 3: Backend Python Hydra Config Generation Updates

**Context**: Use this to adapt the Python script to consume the newly structured subflow tree JSON.

> Act as an MLOps Architect and Python Systems Engineer. Our frontend now produces an updated tree JSON payload containing flattened/unrolled 'SubFlow' definitions.
>
> Here is the current python backend compiler script (`converted/src/convert.py`) which parses strings via `ast.literal_eval` and structures Hydra configs:
> [PASTE YOUR convert.py CONTENT HERE]
>
> Tasks:
>
> 1. Update this compiler script to correctly handle the new SubFlow blocks exported from the frontend tree payload.
> 2. Ensure parameters inside the subflow (like iterations or internal dimensions) are successfully processed and written into structured, modular Hydra config YAML files inside the correct targets.
> 3. Implement explicit type checking and descriptive validation errors during parsing so that malformed parameter boundaries fail-fast on compilation rather than runtime.

### 🗣️ Prompt 4: Mock Viva / Professor Defense Interrogation

**Context**: Use this at the end of the day to practice defending your architecture against a picky examiner.

> Act as a rigorous, pedantic Full Professor teaching an Advanced Software Engineering Master's course at Sapienza University of Rome. You are examining my visual Neural Network modeling DSL project.
>
> I am presenting my architecture:
>
> - Svelte 5 / Svelte Flow frontend handling state reactively via `$state.raw`.
> - A formal UML meta-model specifying syntax constraints.
> - An AST-like compilation engine (`nnTree.ts`) mapping graphs to structural configurations.
> - A Python generation pipeline converting payloads to Hydra configs + a PyTorch Lightning runtime (`net/base.py`) which dynamically sorts execution order using a BFS topological sweep based on in-degree tracking.
>
> Crucial Architectural Compromise Found:
> In my frontend, I have an explicit `Flatten` module stereotype node. However, in `net/base.py`, I have added a defensive feature where the Python forward pass checks shapes and automatically triggers a `torch.flatten()` before any `Linear` layer if the tensor dimension exceeds 2, bypassing the explicit canvas node necessity.
>
> Interrogate me:
>
> 1. Ask me 3 highly technical questions about the correctness of my compilation pipeline and the BFS runtime overhead.
> 2. Attack the explicit `Flatten` vs implicit auto-flattening paradox as a classic 'Leaky Abstraction' flaw that violates DSL language purity.
> Provide the harsh questions first, and then provide a set of strategic, enterprise-grade engineering justifications I can use to answer them perfectly.
