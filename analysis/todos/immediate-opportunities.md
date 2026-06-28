# NNModelling — Immediate Opportunities

These are the most natural next steps for the project — improvements that follow directly from the current implementation, fix known issues, or complete partially-built features. Each opportunity is concrete, scoped, and independently valuable.

---

## 1. Fix the Duplicate Source Handle Bug

### Why It Fits the Project

`CustomNode.svelte` renders two `<Handle type="source">` elements stacked at `Position.Bottom`. The first is conditionally hidden for loss nodes (`{#if !data.isLoss}`), but the second is unconditional — meaning **every node renders at least one source handle, and non-loss nodes render two stacked on top of each other**. While Svelte Flow may deduplicate overlapping handles visually, this is undefined behavior and could cause subtle bugs with edge attachment or future Svelte Flow versions.

### Expected Value

- Eliminates the single most obvious bug in the codebase
- Prevents potential edge attachment ambiguity
- Simplifies the component (1 handle instead of 2)

### Estimated Implementation Complexity

**Trivial** — change lines 71-74 of `CustomNode.svelte`: remove the conditional block and keep only the unconditional `<Handle>`. Loss nodes already have no source handle because the unconditional one renders for all nodes; this should be changed so loss nodes truly have no source handle.

```svelte
<!-- Before (buggy): -->
{#if !data.isLoss}
  <Handle type="source" position={Position.Bottom} {isConnectable} />
{/if}
<Handle type="source" position={Position.Bottom} {isConnectable} />

<!-- After (fixed): -->
{#if !data.isLoss}
  <Handle type="source" position={Position.Bottom} {isConnectable} />
{/if}
```

### Suggested Implementation Order

**1st** — do this first as a warm-up before anything else. It's a one-line fix with zero risk.

---

## 2. Semantic Connection Validation (Shape Compatibility Warnings)

### Why It Fits the Project

The current `checkValidConnection()` validates only handle occupancy — "is this target handle already taken?" It does not check whether the connection is semantically valid. A user can connect a Conv2d layer (4D output) to a Linear layer (expects 2D input) without any warning. This silently produces a model that will fail at runtime with an opaque PyTorch error.

The project already has all the information needed for basic validation: stereotype metadata (is this a convolution? a linear layer? a flatten?), parameter values (in_channels, out_features), and topological context. Adding shape-aware warnings would dramatically improve the editing experience and prevent the most common class of user errors.

### Expected Value

- **Catches the #1 source of silent errors** in hand-designed architectures
- Educational — teaches users about tensor shape constraints through warnings
- Low false-positive rate if designed as warnings, not hard blocks
- Foundation for the longer-term Shape-Aware Graph Engine

### Estimated Implementation Complexity

**Medium** — requires:
1. A shape "signature" per stereotype (e.g., `Conv2d` → `[B, C, H, W] → [B, C', H', W']`)
2. A `suggestShapeCompatibility(sourceStereo, sourceParams, targetStereo, targetParams)` function
3. Integration into `checkValidConnection()` as a warning (not a blocking error — allow the connection but show a visual indicator)

Start with the most common mismatches: Conv→Linear without Flatten, Linear→Conv without reshape, dimensionality mismatches. Expand incrementally.

### Suggested Implementation Order

**2nd** — implement after the handle bug fix. Adds immediate user value and is standalone.

---

## 3. Stereotype Catalog Expansion (Common Missing Layers)

### Why It Fits the Project

12 of 35 existing stereotypes are unused in any example diagram (Dropout, BatchNorm1d, BatchNorm2d, BCEWithLogitsLoss, BCELoss, Einsum, MaskedScaledDotProduct, Transformer, TransformerEncoderLayer, TransformerDecoderLayer, AvgPool2d, MultiheadAttention). Several common layers used in modern architectures are missing entirely:

- **GELU** — standard in transformers (GPT, BERT, ViT). Currently only ReLU, Tanh, Sigmoid, Softmax.
- **LeakyReLU** — prevents dying ReLU problem.
- **Conv1d** — natural counterpart to Conv2d for 1D sequence/temporal data.
- **AdaptiveAvgPool2d** — used in ResNet, EfficientNet (outputs fixed spatial size regardless of input).
- **Dropout2d** — channel-wise dropout for conv layers.
- **GroupNorm** — used in ViT, Stable Diffusion. Alternative to BatchNorm for small batch sizes.

Adding these fills obvious gaps and makes the stereotype catalog feel complete for modern architecture design.

### Expected Value

- Users building transformers can use GELU without needing a custom stereotype
- Conv1d unlocks temporal/audio/text CNN architectures
- AdaptiveAvgPool2d enables ResNet-style global pooling
- Small effort per stereotype (~1-minute JSON file creation), high cumulative value

### Estimated Implementation Complexity

**Low** — each stereotype is a ~10-line JSON file. No code changes needed for standard `torch.nn.*` modules (they already work through the existing `build_layer_config` pipeline). BatchNorm1d and Conv2d JSONs serve as templates.

### Suggested Implementation Order

**3rd** — can be done in parallel with other items since it's pure data entry.

---

## 4. Training Configuration from the Visual Editor

### Why It Fits the Project

Currently, training hyperparameters (optimizer, learning rate, batch size, epochs, early stopping patience) are hardcoded in `convert.py` or set via CLI flags. The visual editor has no awareness of training configuration. This creates a disconnect: you design the architecture visually, then switch to the command line to configure training.

The Sidebar already handles per-node parameters. Adding a "Training Config" panel (or a dedicated node type) would close the loop, making the editor the single interface for both architecture and training.

### Expected Value

- Complete the "design in editor → train" workflow without leaving the UI
- Users don't need to know Hydra YAML syntax to configure training
- Enables saving/loading complete experiment configurations (architecture + training params in one JSON)
- Natural extension of the existing Sidebar form pattern

### Estimated Implementation Complexity

**Medium** — requires:
1. A `TrainingConfig` section in the Sidebar (or a modal/dialog) with fields for: optimizer choice (Adam/SGD/AdamW), learning rate, batch size, max epochs, early stopping patience, seed
2. Extending `exportToJson()` and `importFromJson()` to include training config
3. Extending `convert.py` to read training config from the JSON (currently hardcoded)
4. Optional: a `TrainingConfig` node type on the canvas (like a global settings node)

The `convert.py` changes are straightforward — instead of hardcoding `Adam(lr=0.001)`, it reads from JSON. The frontend work is standard Svelte form building.

### Suggested Implementation Order

**4th** — high user value, moderate effort. Depends on understanding the `convert.py` config generation.

---

## 5. Subflow Save/Load (Reusable Components)

### Why It Fits the Project

The DSL requirements (`reqs.md` §1.4.3) specify that submodels should be individually saveable to disk. Currently, subflows exist only within a diagram — there's no way to extract a subflow as a standalone component file or import one into another diagram.

The data model is already ready for this: subflows are self-contained graph fragments with their own `nodes`, `edges`, `parentId` relationships. Exporting a subflow means extracting its node+edge subgraph and saving as a separate JSON file. Importing means loading that JSON and grafting it into the current diagram.

### Expected Value

- Users can build a library of reusable components (ResNet blocks, attention heads, encoder stacks)
- Enables "component marketplace" as a long-term direction
- Reduces repetitive diagram construction — design once, reuse many times
- Completes a requirement from the original DSL spec

### Estimated Implementation Complexity

**Medium** — requires:
1. **Export**: `diagram.exportSubflow(subflowId)` → extracts the subgraph (children + edges), normalizes coordinates, saves as standalone JSON (a new format: `SubflowComponent` with metadata like name, description, input/output handle count)
2. **Import**: `diagram.importSubflow(json, x, y)` → loads a component JSON, translates coordinates, inserts into current diagram with new UUIDs
3. **UI**: "Save Subflow" button in toolbar (when a subflow is selected), "Load Component" button that opens a file picker
4. **Directory**: `Components/` directory parallel to `Stereotypes/` for user-saved components

The trickiest part is handling handle mapping — an imported subflow needs to expose its internal Input node as the subflow's input, and its internal Loss/output nodes as the subflow's output. The current subflow model already has `entryNode` for this purpose.

### Suggested Implementation Order

**5th** — unlocks component reuse, which amplifies the value of every other feature. Depends on the subflow compilation pipeline being stable (it is).

---

## 6. CI Pipeline (GitHub Actions)

### Why It Fits the Project

The project has **no CI/CD at all**. A 76-test frontend test suite and a 103-test Python test suite that only run when manually executed. Every PR, every push, risk introducing regressions that go undetected until the next manual test run.

The TODO.md explicitly lists CI as pending. The project is currently a solo effort, but CI is even more important for solo developers — it's the only safety net when you're the only one reviewing code.

### Expected Value

- Catches regressions automatically on every push
- Ensures both frontend and backend tests pass before merging
- Enables confident refactoring
- Standard open-source practice — signals project maturity

### Estimated Implementation Complexity

**Low** — a simple GitHub Actions workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - run: cd front-end && pnpm install && pnpm test
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
      - run: cd converted && uv sync && uv run pytest src/tests/
```

The frontend tests don't need a browser (pure TS, no DOM), and the Python tests instantiate real models but on CPU only (fast enough for CI). Both suites complete in under 30 seconds based on their size.

### Suggested Implementation Order

**6th** — very low effort, very high value. Should be done before any of the larger items above to ensure they don't break existing tests.

---

## 7. HorizontalRepeat Join Type Configurability

### Why It Fits the Project

`ops.HorizontalRepeat` currently hardcodes the join operation to `concat on dim=-1`. The source code itself documents this limitation:

> *"To use a different join (add, mean, max, etc.), modify forward() or create a new op — HorizontalRepeat does not expose join_type as a parameter."*

The project already has `ops.Addition` (sum) and `ops.Concat` (concatenation) as standalone join types. Extending `HorizontalRepeat` to accept a `join_type` parameter (e.g., `"concat"`, `"add"`, `"mean"`) would make it a truly general parallel-execution primitive. An `n=4` HorizontalRepeat with `join_type="add"` would produce `[batch, d]` output by summing the 4 heads, rather than concatenating to `[batch, 4*d]`.

### Expected Value

- Unlocks use cases beyond multi-head attention (e.g., ensemble averaging, multi-path aggregation)
- Makes HorizontalRepeat a proper general-purpose behavioral stereotype
- Small code change, significant capability expansion
- Already planned in TODO.md

### Estimated Implementation Complexity

**Low** — changes in three places:
1. **`HorizontalRepeat.json` stereotype** — add `join_type` param (default `"concat"`, options: `"add"`, `"mean"`, `"max"`)
2. **`ops/horizontal_repeat.py`** — add `join_type` init param, branch in `forward()`: if `"concat"` (current behavior), if `"add"` (sum heads), if `"mean"` (mean heads), if `"max"` (element-wise max)
3. **Tests** — add test cases for each join_type

### Suggested Implementation Order

**7th** — small, self-contained, unlocks new diagram patterns.

---

## 8. Error Boundaries for Conversion Pipeline

### Why It Fits the Project

When `new NNTree(diagram)` encounters an invalid graph (no Input node, cycle in subflow, etc.), it throws an unhandled exception. The `handleConversion()` in `FlowCanvas.svelte` does not catch this — the error propagates to the browser console, and the user sees nothing except that the "Convert" button did nothing.

This is a poor UX for a visual tool. The user should see a clear error message explaining what's wrong and how to fix it.

### Expected Value

- Dramatically better user experience for diagram debugging
- Users can fix diagram errors without opening the browser console
- Prevents user frustration with silent failures
- Foundation for richer error reporting (highlighting problematic nodes on the canvas)

### Estimated Implementation Complexity

**Low** — wrap `new NNTree(diagram)` in a try/catch in `handleConversion()`, display errors in a toast/notification or in the Sidebar as an error panel. The NNTree already produces descriptive error messages (e.g., "Expected exactly one input node", "Subflow contains a cycle").

A simple implementation: add a `lastError: string | null` state to `FlowCanvas`, display it as a red banner below the toolbar when non-null, clear on next successful operation.

### Suggested Implementation Order

**8th** — quick win for UX, very low effort.

---

## 9. Empty Subflow Handling

### Why It Fits the Project

When `compileSubflowGraph()` encounters a subflow with no internal nodes, it `console.warn`s and returns `{ entryNode: "", nodes: {} }`. This creates downstream issues: an empty `entryNode` string that `ops.Subflow.__init__` treats as the first node to execute, and empty `nodes` that instantiate nothing. The Python side silently passes through input.

This is a design edge case that should either be properly handled (subflows with 0 internal nodes are valid identity passthroughs) or explicitly rejected (subflows must contain at least one node).

### Expected Value

- Removes ambiguity in the compilation pipeline
- Prevents mysterious silent passthroughs
- Better user feedback when they create an empty subflow

### Estimated Implementation Complexity

**Low** — decide on the desired behavior and implement consistently:
- **Option A (recommended):** Reject empty subflows at compile time with a clear error. Require at least one internal node.
- **Option B:** Accept empty subflows as explicit identity operations. Set `entryNode` to a sentinel value, and have `ops.Subflow` detect it and passthrough.

Either way, the `console.warn` should become a proper error or validated identity path.

### Suggested Implementation Order

**9th** — small fix, removes a source of potential silent bugs.

---

## 10. Debounced Sidebar Parameter Updates

### Why It Fits the Project

`Sidebar.svelte` calls `diagram.updateModule()` on every `oninput` event. This means every keystroke in a parameter field triggers a full Svelte Flow re-render (node position, handles, edges all re-evaluate). For small diagrams this is imperceptible, but for large diagrams (50+ nodes, or complex subflow hierarchies) it could cause visible lag.

The fix is straightforward: debounce the update by 300-500ms, or switch from `oninput` to `onchange` (update only on blur). The latter is simpler and avoids the debounce complexity, but changes the interaction model slightly (params update when you click away, not live).

### Expected Value

- Prevents potential performance issues as diagrams grow
- Standard practice for form-to-canvas synchronization
- No visible UX regression for normal-sized diagrams

### Estimated Implementation Complexity

**Trivial** — change `oninput={handleLiveUpdate}` to `onchange={handleLiveUpdate}` in Sidebar.svelte. Or add a simple debounce wrapper.

### Suggested Implementation Order

**10th** — trivial preventative fix.

---

## 11. Front-End README Rewrite

### Why It Fits the Project

`front-end/README.md` is the auto-generated Vite template README — it tells users to run `npx degit xyflow/vite-svelte-flow-template`, which is not how this project works. This is confusing for new contributors.

### Expected Value

- Clear onboarding for new developers
- Documents the front-end architecture at a high level
- Explains how to add new stereotypes, run tests, understand the conversion pipeline

### Estimated Implementation Complexity

**Trivial** — replace the template content with project-specific documentation: setup instructions, architecture overview, development workflow, test commands, adding stereotypes guide.

### Suggested Implementation Order

**11th** — documentation, can be done anytime.

---

## 12. Remove Dead TODO Comment

### Why It Fits the Project

`FlowCanvas.svelte:49` contains:
```typescript
// TODO: serve questo codice? let isNodeSelected = $derived(selectedNodes.length === 1);
```

The TODO.md analysis explicitly identifies this as dead code to remove, and it's listed in the project's own TODO tracking. The derived value `isNodeSelected` is indeed never used in the component.

### Expected Value

- Cleaner codebase
- One less item on the TODO list

### Estimated Implementation Complexity

**Trivial** — delete one comment line.

### Suggested Implementation Order

**12th** — do it when touching FlowCanvas.svelte for any other change.

---

## 13. Graph Cycle Detection Improvements

### Why It Fits the Project

The current cycle detection in `nnTree.ts` catches cycles at the graph level (throws) and subflow level (throws), but only uses a simple recursive-with-visited-set approach. Several edge cases are untested:

- Self-referencing nodes (node connects to itself)
- Cycles crossing subflow boundaries (a node inside a subflow connecting to a node outside, and vice versa)
- Multi-node cycles (A→B→C→A) — likely caught by the existing algorithm but not explicitly tested

Adding these edge case tests and hardening the detection would prevent subtle bugs when users create complex nested architectures.

### Expected Value

- Prevents silent infinite recursion in the conversion pipeline
- Better error messages for specific cycle patterns
- Confidence that the topological sort is correct for all graph topologies

### Estimated Implementation Complexity

**Low** — primarily test additions (5-8 new test cases) plus potentially small refinements to the detection logic. The existing `visited`-and-recursion approach likely handles most cases; the main work is proving it through tests.

### Suggested Implementation Order

**13th** — pairs well with item #8 (error boundaries for conversion). Test-driven: write the failing tests, verify they're caught, then improve error messages.

---

## 14. Python Test Coverage: Dataset Classes and Inference

### Why It Fits the Project

The Python test suite has excellent coverage of ops (36 tests), base.py (21 tests), convert.py (35 tests), and integration (11 tests). But two important areas have **zero tests**:

1. **Dataset classes** — `mnist.py`, `autoencoder_mnist.py`, `enron_spam.py`, and `ds.py` are untested. If the MNIST download URL changes, or the EnronSpam dataset schema changes, tests won't catch it.
2. **Inference script** — `infer.py` is untested. The `--output` and `--image-dir` flags, image tensor detection logic, and error handling paths are all unverified.

### Expected Value

- Prevents regressions in data loading and inference
- Catches upstream dataset changes before users encounter them
- Enables confident refactoring of dataset classes

### Estimated Implementation Complexity

**Medium** — dataset tests require either:
- Mocking torchvision/HF datasets (fast, but fragile to API changes)
- Using small real data samples (reliable, but requires network access in CI)
- `infer.py` tests require a trained model artifact (already saved as `weights.pt` in the repo)

A pragmatic approach: test `ds.py` abstract interface, mock the dataset downloads for unit tests, and add 1-2 integration tests that use the actual datasets if network is available.

### Suggested Implementation Order

**14th** — important for long-term maintainability. Can be done alongside other items.

---

## 15. Front-End Component Tests (at Minimum: Smoke Tests)

### Why It Fits the Project

The front-end test suite covers `nnTree.ts` (66 tests) and `utils.ts` (9 tests) thoroughly, but has **zero component tests**. No tests for `Diagram.svelte.ts`, `Sidebar.svelte`, `FlowCanvas.svelte`, `CustomNode.svelte`, `JoinNode.svelte`, or `SubflowNode.svelte`.

At minimum, the `Diagram.svelte.ts` class (the central state manager) should have unit tests for its public methods: `addModule()`, `addJoinNode()`, `deleteNodes()`, `importFromJson()`/`exportToJson()`, `toggleSubflow()`. These are pure TypeScript with no DOM dependency — testable with the same `stubWindow()` pattern already used in `nnTree.test.ts`.

### Expected Value

- Prevents regressions in the most critical state management code
- Enables confident refactoring of the Diagram class
- Catches serialization/deserialization bugs (import/export roundtrip)

### Estimated Implementation Complexity

**Medium** — `Diagram.svelte.ts` is 346 lines with ~12 public methods. Testing each method with a few cases would be 20-30 tests. The existing `stubWindow()` and `node()`/`edge()` helpers make this straightforward.

Svelte component tests (testing rendered DOM) would require `@testing-library/svelte` and are significantly more complex. Skip those for now and focus on the pure-TypeScript `Diagram` class.

### Suggested Implementation Order

**15th** — important for maintainability. Start with `addModule`, `deleteNodes`, and `importFromJson`/`exportToJson` as the most critical methods.

---

## Implementation Priority Summary

| # | Opportunity | Complexity | Impact | Dependencies |
|---|-----------|-----------|--------|-------------|
| 1 | Fix duplicate Handle bug | Trivial | Bug fix | None |
| 2 | Semantic connection validation | Medium | High UX | None |
| 3 | Stereotype catalog expansion | Low | Medium UX | None |
| 4 | Training config from editor | Medium | High UX | convert.py changes |
| 5 | Subflow save/load | Medium | High UX | Subflow compilation |
| 6 | CI pipeline | Low | Dev infra | None |
| 7 | HorizontalRepeat join type | Low | Capability | None |
| 8 | Error boundaries for conversion | Low | UX | None |
| 9 | Empty subflow handling | Low | Bug fix | None |
| 10 | Debounced sidebar updates | Trivial | Performance | None |
| 11 | Front-end README rewrite | Trivial | Docs | None |
| 12 | Remove dead TODO comment | Trivial | Cleanup | None |
| 13 | Cycle detection hardening | Low | Robustness | None |
| 14 | Dataset/inference tests | Medium | Test infra | CI pipeline |
| 15 | Diagram class tests | Medium | Test infra | None |

**Recommended execution order**: 1 → 8 → 6 → 2 → 10 → 12 → 9 → 13 → 7 → 3 → 15 → 14 → 4 → 5 → 11

This order fixes bugs first (1, 8, 9), establishes CI safety net (6), adds immediate user value (2, 10), cleans up (12), hardens (13), expands capabilities (7, 3), strengthens test coverage (15, 14), and finally adds major features (4, 5) with documentation (11).
