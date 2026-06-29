
# Goal

Introduce a static tensor type system into the NNModelling DSL capable of verifying tensor shapes and dtypes during visual editing and front-end compilation, preventing runtime PyTorch shape/type errors.

The implementation should integrate with the existing architecture without changing the current NNTree compilation pipeline.

---

# Repository Context

Before implementing anything, analyze the following files:

* `front-end/src/conversion/nnTree.ts`
* `front-end/src/utils.ts`
* `front-end/src/stereotype.ts`

Also inspect how stereotypes are loaded from:

* `Stereotypes/**/*.json`

The goal is to understand:

1. How graph compilation currently works.
2. How stereotypes are represented.
3. Where type inference should hook into the pipeline.
4. How connection validation currently works.

Do **not** start implementing before explaining the architecture.

---

# Overall Design Goal

The type system should be extensible enough to eventually support:

* Sequential modules
* Multi-input joins
* Convolutional layers
* Attention operators
* Flatten/Reshape operations
* Subflows
* Generic symbolic dimensions
* Future compile-time optimizations

Avoid implementing ad-hoc shape checks for individual modules.

Instead, design a reusable type inference framework.

---

# Deliverables

## Phase 1 — Architectural Analysis

Explain:

* Where the TypeEngine should live.
* How it integrates with `NNTree`.
* How stereotype loading should evolve.
* How inferred types propagate through the graph.
* How errors should be surfaced to the editor.
* Which data should be stored inside nodes versus stereotypes.

If additional files are required, justify them before creating them.

---

## Phase 2 — Type Model

Design TypeScript interfaces for:

```ts
TensorType
TensorShape
ShapeDimension
TypeSignature
ModuleTypeSignature
JoinTypeSignature
ShapePattern
TypeEnvironment
TypeError
```

Please add them in a new file in `front-end/conversion/tensortypes.ts`

The model should support:

* symbolic dimensions (`B`, `T`, `H`, `W`, ...)
* wildcard dimensions (`...`)
* unknown dimensions
* dtype propagation
* parameter references
* future symbolic constraints

The design should not assume every dimension is numeric.

---

## Phase 3 — Shape Representation

Do **not** model tensor shapes as plain arrays of strings.

Instead, design an intermediate representation capable of expressing:

* symbolic dimensions
* parameter references
* wildcard dimensions
* future computed dimensions

For example, instead of:

```json
{
  "shape": ["...", "params.in_features"]
}
```

design a representation that can distinguish:

* symbolic variables
* parameter references
* wildcard/rest dimensions
* future expressions

The representation should make it possible to support operations such as:

* Conv2d output size
* Flatten
* Reshape
* Concat
* MatMul
* Einsum

without redesigning the schema.

---

## Phase 4 — Type Signatures

Extend stereotype JSON files with a declarative `type_signature`.

Example:

```json
{
  "type_signature": {
    "kind": "Linear",
    "input": [
      "$B",
      "...",
      "params.in_features"
    ],
    "output": [
      "$B",
      "...",
      "params.out_features"
    ]
  }
}
```

The stereotype should remain **declarative**.

Avoid embedding executable expressions inside the JSON.

The JSON should describe **what** the module expects, not **how** inference is computed.

---

## Phase 5 — Type Engine

Create:

```text
front-end/src/conversion/typeEngine.ts
```

The `TypeEngine` must be completely **data-driven**.

It must not contain module-specific logic (`Linear`, `Conv2d`, `Flatten`, `Concat`, etc.) or hardcoded conditionals based on stereotype names or categories.

Instead, the engine should interpret the `type_signature` defined in each stereotype and implement a generic inference algorithm capable of:

* matching input tensor patterns
* binding symbolic dimensions
* resolving parameter references
* propagating dtypes
* validating declared constraints
* computing output tensor types

The engine should know **how to interpret the type system**, but never **what a specific module does**.

Adding support for a new module should require modifying **only its stereotype JSON**, without changing the TypeScript implementation whenever possible.

Initially implement only the subset of the type system required to support:

* `Input`
* `Linear`
* `ReLU`

Design the architecture so that future modules (e.g. `Conv2d`, `Flatten`, `Concat`, `MatMul`, `Attention`, `SubFlow`) can be supported by extending the declarative `type_signature` schema rather than the inference engine itself

---

## Phase 6 — Unit Tests

Create:

```
front-end/src/__tests__/typeEngine.test.ts
```

Follow the same testing style as `nnTree.test.ts`.

Reuse the existing `Diagram` helpers whenever possible.

Implement the following Tier-0 tests:

* Input → Linear
* Input → Linear → ReLU
* Linear shape mismatch
* dtype mismatch

---

# Constraints

* Preserve the existing architecture.
* Avoid breaking API changes.
* Do not refactor unrelated files.
* Keep the implementation incremental.
* Prefer reusable abstractions over module-specific logic.
* Clearly mark unsupported modules with TODO comments.

---

# Expected Output

Produce the answer in the following order:

1. Architectural Analysis
2. Comparison of at least **three** possible architectures
3. Justification of the selected approach
4. File structure changes
5. Complete TypeScript implementation
6. Unit tests
7. Remaining TODOs

---

# Important

Before writing any code:

1. Compare at least three architectural approaches.
2. Explain the trade-offs of each.
3. Justify the chosen solution.
4. State explicitly any assumptions instead of guessing.

Focus on designing a type system that remains maintainable as the DSL grows to include more tensor operators and graph constructs.
