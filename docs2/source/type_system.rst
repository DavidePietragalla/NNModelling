Tensor Type System
==================

NNModelling includes a **static tensor type system** that verifies tensor shapes
and dtypes during visual editing and front-end compilation. The type system
catches shape mismatches (e.g. connecting a ``Linear`` layer expecting 784 input
features to a ``Conv2d`` producing a 3D tensor) in the editor, before PyTorch
raises a runtime dimension error during training.

The type system is a separate verification pass that operates on the diagram
graph and returns type annotations and errors. It is **data-driven**: all
module-specific logic is declared in ``type_signature`` fields inside stereotype
JSON files. Adding a new module requires only a JSON change, never a TypeScript
modification.

Architecture
------------

The type engine is an **independent module** (Architecture C in the design docs)
that takes a ``Diagram`` and returns a ``TypeResult``. It has no knowledge of
Svelte, the UI, or the Python backend. This design was chosen after comparing
three alternatives:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - Criterion
     - A — Embedded in NNTree
     - B — Post-NNTree Verification
     - C — Independent Engine
   * - Separation of concerns
     - ❌ Low
     - ✅ High
     - ✅ High
   * - Real-time editor feedback
     - ❌ No
     - ❌ No (post-hoc)
     - ✅ Yes
   * - Testability
     - ❌ Hard
     - ✅ Easy
     - ✅ Easy
   * - Data-driven (no code changes for new modules)
     - ❌ Mixed
     - ✅ Yes
     - ✅ Yes
   * - Evolvability
     - ❌ Poor
     - ✅ Good
     - ✅ Best

Architecture C was selected: the ``TypeEngine`` is a pure function
(``infer(diagram) → TypeResult``) invoked from multiple call sites — the
editor (real-time), the NNTree compiler (embed types), and unit tests.

The engine is invoked from three points:

.. code-block:: text

   TypeEngine.infer(diagram)
        │
        ├── FlowCanvas.svelte (on edge connect/disconnect, diagram load)
        ├── Sidebar.svelte (on parameter change, debounced ~300ms)
        └── typeEngine.test.ts (50+ unit tests)

Type Model
----------

The type model is defined in ``front-end/src/conversion/tensortypes.ts``.
This file contains only interfaces and type aliases, no runtime logic.

Tensor Types
~~~~~~~~~~~~

A tensor type :math:`\tau` is a pair consisting of a shape and a data type:

.. math::

   \tau ::= \text{Tensor}(\sigma, \delta)

where :math:`\sigma` is a tensor shape and :math:`\delta` is a tensor data type
(e.g. ``float32``, ``int64``).

Shape Dimensions
~~~~~~~~~~~~~~~~

A shape :math:`\sigma` is a finite sequence of dimensions
:math:`d_1, d_2, \ldots, d_n`. Each dimension :math:`d` belongs to one of the
following categories:

.. math::

   d ::= c \mid x \mid p \mid *

.. list-table::
   :header-rows: 1
   :widths: 15 15 35 35

   * - Kind
     - Notation
     - Meaning
     - Example
   * - ``const``
     - :math:`c \in \mathbb{N}`
     - A literal integer dimension
     - ``{ kind: "const", value: 784 }``
   * - ``symbolic``
     - :math:`x \in \mathcal{X}`
     - A named dimension variable (e.g. batch size :math:`B`, sequence length :math:`L`)
     - ``{ kind: "symbolic", name: "B" }``
   * - ``param_ref``
     - :math:`p \in \mathcal{P}`
     - References a node parameter (resolved at inference time)
     - ``{ kind: "param_ref", name: "in_features" }``
   * - ``wildcard``
     - :math:`*`
     - Matches zero or more arbitrary dimensions
     - ``{ kind: "wildcard" }``
   * - ``computed``
     - :math:`f(\vec{a})`
     - A dimension computed by formula (e.g. ``conv2d_hw``)
     - ``{ kind: "computed", formula: "conv2d_hw", args: ["$H", "kernel_size", ...] }``

**Notation rule**: In JSON, symbolic dimension names start with ``$`` (e.g.
``"$B"``, ``"$H"``, ``"$W"``). This distinguishes them from ``param_ref`` names.
When loaded into ``ShapeDimPattern``, the ``$`` is stripped: ``"$B"`` becomes
``{ kind: 'symbolic', name: 'B' }``. Parameter references never have ``$``.

Type Signatures
~~~~~~~~~~~~~~~

Every stereotype can optionally declare a ``type_signature`` field describing
its tensor shape contract:

.. code-block:: typescript

   interface TypeSignature {
     kind: 'module' | 'join' | 'subflow';
     input: ShapePattern | ShapePattern[];
     output: ShapePattern;
     dtype?: { input?: DType; output?: DType };
     constraints?: {
       concat?: { dim: string };  // "params.dim" — for Concat joins
     };
   }

Key concepts:

* **Input pattern**: the shape(s) the node expects to receive. A single
  ``ShapePattern`` for modules; an array of ``ShapePattern`` for joins (one
  per input handle).
* **Output pattern**: the shape the node produces.
* **Dtype constraints**: optional restrictions on input/output dtypes.
* **Concat constraint**: for ``Concat`` joins, specifies which dimension is
  concatenated (as a parameter reference like ``"params.dim"``).

Typing Context
~~~~~~~~~~~~~~

The typing context :math:`\Gamma` is a partial mapping from symbolic dimension
names to their resolved values:

.. math::

   \Gamma ::= \{ x_1 \mapsto d_1,\; x_2 \mapsto d_2,\; \ldots \}

The context is populated incrementally during type inference as symbolic
dimensions are bound to concrete values. It also carries dtype information and
maintains a mapping from node identifiers to their inferred tensor types.

Typing Judgments
~~~~~~~~~~~~~~~~

The central judgment form for node-level type inference is:

.. math::

   \Gamma, P \vdash M : (\tau_{\text{in}} \rightarrow \tau_{\text{out}})

meaning: "in context :math:`\Gamma` with parameter values :math:`P`, module
:math:`M` maps input type :math:`\tau_{\text{in}}` to output type
:math:`\tau_{\text{out}}`."

For graph-level inference:

.. math::

   \Gamma \vdash G : \Gamma'

meaning: "graph :math:`G` is well-typed, producing the extended environment
:math:`\Gamma'` containing type annotations for every node."

Type Result & Errors
~~~~~~~~~~~~~~~~~~~~

.. code-block:: typescript

   interface TypeResult {
     ok: boolean;
     annotations: Map<string, NodeTypeAnnotation>;
     errors: TypeError[];
   }

   interface TypeError {
     nodeId: string;
     message: string;
     severity: 'error' | 'warning';
   }

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

Parameter resolution returns a discriminated union (``ParamResolution``)
rather than silently treating invalid values as unset:

.. code-block:: typescript

   type ParamResolution =
     | { status: 'unset' }                           // "None" / missing
     | { status: 'invalid'; value: string }          // "cazz", "hello"
     | { status: 'resolved'; value: number };         // 784

Non-numeric parameter values (e.g. "cazz" for ``in_features``) generate
type errors instead of being silently treated as unset.

Stereotype Type Signatures
--------------------------

Each stereotype JSON can declare a ``type_signature`` field. Here are the
canonical examples:

Input Node
~~~~~~~~~~

The Input node is a source in the computation graph. It produces a tensor
whose last dimension is determined by its ``out_features`` parameter:

.. code-block:: json

   {
     "category": "Input",
     "pythonClassName": "None",
     "params": {
       "out_features": { "type": "int", "default": "784" }
     },
     "type_signature": {
       "kind": "module",
       "input": [],
       "output": [
         { "kind": "symbolic", "name": "$B" },
         { "kind": "param_ref", "name": "out_features" }
       ],
       "dtype": { "output": "float32" }
     }
   }

.. math::

   \frac{
     \text{stereotype}(v) = \text{Input}
     \qquad
     P = \text{params}(v)
   }{
     \Gamma \vdash v :
     \text{Tensor}((B,\, P.\text{out\_features}),\, \text{float32})
   }

where :math:`B` is a fresh symbolic dimension variable introduced into
:math:`\Gamma`.

Linear Layer
~~~~~~~~~~~~

A ``Linear`` layer applies an affine transformation to the last dimension
of its input:

.. code-block:: json

   {
     "category": "Layer",
     "pythonClassName": "nn.Linear",
     "params": {
       "in_features": { "type": "int", "default": "Undefined", "position": "top" },
       "out_features": { "type": "int", "default": "Undefined", "position": "bottom" },
       "bias": { "type": "bool", "default": "True" }
     },
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

.. math::

   \frac{
     \Gamma \vdash x :
     \text{Tensor}((B, \alpha_1, \ldots, \alpha_k, F), \delta)
     \qquad
     F = P.\text{in\_features}
   }{
     \Gamma \vdash \text{Linear}(P)(x) :
     \text{Tensor}((B, \alpha_1, \ldots, \alpha_k, P.\text{out\_features}), \delta)
   }

where :math:`\alpha_1, \ldots, \alpha_k` are intermediate dimensions matched by
the wildcard pattern and carried forward unchanged. The wildcard in ``[B, *, in_features]``
can match zero or more intermediate dimensions.

ReLU Activation
~~~~~~~~~~~~~~~

Activation functions are shape-preserving and dtype-preserving:

.. code-block:: json

   {
     "category": "Layer",
     "pythonClassName": "nn.ReLU",
     "params": {
       "inplace": { "type": "bool", "default": "False" }
     },
     "type_signature": {
       "kind": "module",
       "input": [{ "kind": "wildcard" }],
       "output": [{ "kind": "wildcard" }]
     }
   }

.. math::

   \frac{
     \Gamma \vdash x : \text{Tensor}(\sigma, \delta)
   }{
     \Gamma \vdash \text{ReLU}(x) : \text{Tensor}(\sigma, \delta)
   }

The same rule applies to all shape-preserving modules: Tanh, Sigmoid, Softmax,
Dropout, BatchNorm1d, BatchNorm2d, LayerNorm.

Embedding
~~~~~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "symbolic", "name": "$L" }
     ],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "symbolic", "name": "$L" },
       { "kind": "param_ref", "name": "embedding_dim" }
     ]
   }

Inference Engine
----------------

The inference engine is implemented in
``front-end/src/conversion/typeEngine.ts``. It is a **pure, data-driven
function** with no hardcoded module names, no category checks, and no Svelte
or DOM dependencies.

Algorithm
~~~~~~~~~

The core inference algorithm operates in two phases:

1. **Constraint Generation**: For each node visited in topological order, the
   engine reads the node's stereotype ``type_signature`` and generates
   constraints by pattern-matching the actual input tensor shape against the
   declared input pattern. This produces bindings (symbolic dimensions are
   bound to concrete values), substitutions (wildcard dimensions capture a
   suffix of the actual shape), and resolutions (parameter references are
   looked up from the node's parameter map).

2. **Constraint Solving**: The engine substitutes bound variables and captured
   wildcards into the output pattern, producing the output tensor type. If any
   constraint is violated (const dimension mismatch, dtype mismatch,
   unresolvable parameter reference), a ``TypeError`` is recorded.

.. code-block:: text

   TypeEngine.infer(diagram):
     1. Build topological order (Kahn's algorithm on top-level nodes)
     2. For each node in order:
        a. Get stereotype and typeSignature
        b. If no typeSignature → warning, skip
        c. Determine input type(s) from predecessor annotations
        d. Call patternMatch(inputShape, inputPattern, params, env)
        e. If match fails → record TypeError, continue
        f. Resolve output: substitute bindings + captured wildcards
        g. Store annotation, update environment
     3. Return { ok, annotations, errors }

Topological Sort
~~~~~~~~~~~~~~~~

The engine uses Kahn's algorithm for topological sorting:

.. code-block:: text

   1. Build adjacency list from edges
   2. Build in-degree map for all top-level nodes
   3. Queue = nodes with in-degree 0
   4. While queue not empty:
      - Dequeue, add to sorted result
      - For each child, decrement in-degree; if 0, enqueue
   5. If sorted.length < total nodes → cycle detected (warning)

Only top-level nodes (``parentId === undefined``) are traversed. Subflow
internals are deferred to Phase 4.

Pattern Matching
~~~~~~~~~~~~~~~~

The pattern matching algorithm walks the input dimensions and pattern elements
with a two-pointer approach:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Kind
     - Behavior
     - Error Condition
   * - ``const``
     - Input dim at this position must equal the constant value
     - ``dimension mismatch: expected 3, got 1``
   * - ``symbolic``
     - If unbound in env: bind to input dim. If bound: unify (must match)
     - ``symbolic $B already bound to 784, cannot unify with 128``
   * - ``param_ref``
     - Resolve parameter from node params, compare against input dim
     - ``parameter in_features has invalid value "cazz", expected a number``
   * - ``wildcard``
     - Consume zero or more input dims (with lookahead for subsequent required elements)
     - (none — always succeeds)
   * - ``computed``
     - Pass-through on input (exact value not validated against input)
     - (none — validated on output side)

**Wildcard lookahead**: The wildcard does not blindly consume all remaining
dims. It computes how many dims are needed for subsequent non-wildcard pattern
elements and preserves them. For example, the pattern ``[B, *, in_features]``
on an input of ``[B, 128, 784]``: the wildcard ``*`` consumes only one dim
(128), leaving the last dim for ``in_features``. If the input were ``[B, 784]``,
the wildcard would consume zero dims.

**Symbolic unification**: When a symbolic name appears in multiple positions
(e.g. ``$K`` in both MatMul input patterns), the engine verifies that all
occurrences bind to the same concrete value. If the same symbol is bound to
conflicting values, a unification error is reported.

Example inference chain:

.. code-block:: text

   Input(out_features=784)
     │  output: [B, 784] (float32)
     ▼
   Linear(in_features=784, out_features=256)
     │  patternMatch([B, 784], [B, *, in_features]) → bindings: {B→B, ?→128}
     │  resolvePattern([B, *, out_features]) → [B, 256]
     ▼
     output: [B, 256] (float32)
     │
     ▼
   ReLU()
     │  patternMatch([B, 256], [*]) → captured: [B, 256]
     │  resolvePattern([*], captured) → [B, 256]
     ▼
     output: [B, 256] (float32)

Computed Dimensions (Phase 2)
-----------------------------

Some modules produce output dimensions that are functions of their input
dimensions and parameters. These are expressed using the ``computed`` dimension
kind.

Supported Formulas
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Formula
     - Definition
     - Used By
   * - ``conv2d_hw``
     - :math:`\left\lfloor \frac{H + 2p - d(k-1) - 1}{s} + 1 \right\rfloor`
     - Conv2d
   * - ``pool2d_hw``
     - :math:`\left\lfloor \frac{H + 2p - k}{s} + 1 \right\rfloor`
     - MaxPool2d, AvgPool2d
   * - ``flatten_prod``
     - :math:`\prod_{i} d_i`
     - Flatten

Where :math:`H` is the input height/width, :math:`k` is kernel size,
:math:`s` is stride, :math:`p` is padding, :math:`d` is dilation.

Conv2d
~~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "param_ref", "name": "in_channels" },
       { "kind": "symbolic", "name": "$H" },
       { "kind": "symbolic", "name": "$W" }
     ],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "param_ref", "name": "out_channels" },
       { "kind": "computed", "formula": "conv2d_hw",
         "args": ["$H", "kernel_size", "stride", "padding", "dilation"] },
       { "kind": "computed", "formula": "conv2d_hw",
         "args": ["$W", "kernel_size", "stride", "padding", "dilation"] }
     ]
   }

MaxPool2d / AvgPool2d
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "symbolic", "name": "$C" },
       { "kind": "symbolic", "name": "$H" },
       { "kind": "symbolic", "name": "$W" }
     ],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "symbolic", "name": "$C" },
       { "kind": "computed", "formula": "pool2d_hw",
         "args": ["$H", "kernel_size", "stride", "padding"] },
       { "kind": "computed", "formula": "pool2d_hw",
         "args": ["$W", "kernel_size", "stride", "padding"] }
     ]
   }

Flatten
~~~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "wildcard" }
     ],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "computed", "formula": "flatten_prod", "args": ["$*"] }
     ]
   }

The ``$*`` argument refers to all dimensions captured by the wildcard — their
product becomes the flattened dimension.

Formula Resolution
~~~~~~~~~~~~~~~~~~

When resolving a ``computed`` dimension during ``resolvePattern``:

1. Each argument in ``args`` is resolved:
   - ``$``-prefixed names → looked up in the symbolic environment
   - ``$*`` → product of all wildcard-captured dimensions
   - Plain names → resolved as parameter references from the node's params
2. If all arguments resolve to concrete numbers, the formula function is called
3. If any argument cannot be resolved (symbolic dimension still unbound), the
   computed dimension remains symbolic (deferred resolution)

Formula resolution is tested directly via unit tests:

.. code-block:: typescript

   // conv2d_hw: floor((32 + 2*1 - 1*(3-1) - 1) / 1 + 1) = 32
   resolveFormula("conv2d_hw", [32, 3, 1, 1, 1])  // → 32

   // pool2d_hw: floor((32 + 0 - 2) / 2 + 1) = 16
   resolveFormula("pool2d_hw", [32, 2, 2, 0])       // → 16

   // flatten_prod: 128 * 7 * 7 = 6272
   resolveFormula("flatten_prod", [128, 7, 7])      // → 6272

Join Type Checking (Phase 3)
----------------------------

Join nodes accept multiple inputs and merge them into a single output. The
type engine validates multi-input shape compatibility.

Addition
~~~~~~~~

Element-wise addition: all inputs must have identical shapes.

.. code-block:: json

   "type_signature": {
     "kind": "join",
     "input": [
       [{ "kind": "wildcard" }],
       [{ "kind": "wildcard" }]
     ],
     "output": [{ "kind": "wildcard" }]
   }

The engine captures dimensions from the first input's wildcard and verifies
that subsequent inputs have identical captured dimensions. If they differ,
an error is reported:

.. code-block:: text

   Addition(B×256, B×128) → Error: "Input 1 dimension 1 mismatch: 256 vs 128"

Concat
~~~~~~

Concatenation along a specified dimension. Other dimensions must match.

.. code-block:: json

   "type_signature": {
     "kind": "join",
     "input": [
       [{ "kind": "wildcard" }],
       [{ "kind": "wildcard" }]
     ],
     "output": [{ "kind": "wildcard" }],
     "constraints": {
       "concat": { "dim": "params.dim" }
     }
   }

The concat dimension is resolved from the node's ``dim`` parameter (default
``-1``, meaning the last dimension). The output on that dimension is the sum
of all input dimensions:

.. math::

   \text{Concat}(\text{dim}=d)(x_1, \ldots, x_n)[d] = \sum_{i=1}^n x_i[d]

For ``(B, 128) + (B, 64)`` on ``dim=-1``:

.. code-block:: text

   Concat input 0: [B, 128], input 1: [B, 64]
   ──────────────────────────────────────────
   Output: [B, 128+64] = [B, 192]

If non-concat dimensions differ, the engine reports an error:

.. code-block:: text

   Concat input 1 dimension 0 mismatch: expected B, got 64

MatMul
~~~~~~

Matrix multiplication. Shape constraint:

.. math::

   (M, K) \times (K, N) \rightarrow (M, N)

.. code-block:: json

   "type_signature": {
     "kind": "join",
     "input": [
       [
         { "kind": "symbolic", "name": "$M" },
         { "kind": "symbolic", "name": "$K" }
       ],
       [
         { "kind": "symbolic", "name": "$K" },
         { "kind": "symbolic", "name": "$N" }
       ]
     ],
     "output": [
       { "kind": "symbolic", "name": "$M" },
       { "kind": "symbolic", "name": "$N" }
     ]
   }

The inner dimension :math:`K` is unified across both inputs. If first input
produces :math:`(32, 64)` and second produces :math:`(128, 64)`, the symbolic
:math:`$K` is bound to 64 from the first input but then cannot unify with 128
from the second — a unification error is reported.

ScaledDotProduct
~~~~~~~~~~~~~~~~

Scaled dot-product attention:

.. math::

   Q(B, H, L, D) \times K(B, H, S, D) \times V(B, H, S, D_{\text{out}}) \rightarrow (B, H, L, D_{\text{out}})

.. code-block:: json

   "type_signature": {
     "kind": "join",
     "input": [
       [
         { "kind": "symbolic", "name": "$B" },
         { "kind": "symbolic", "name": "$H" },
         { "kind": "symbolic", "name": "$L" },
         { "kind": "symbolic", "name": "$D" }
       ],
       [
         { "kind": "symbolic", "name": "$B" },
         { "kind": "symbolic", "name": "$H" },
         { "kind": "symbolic", "name": "$S" },
         { "kind": "symbolic", "name": "$D" }
       ],
       [
         { "kind": "symbolic", "name": "$B" },
         { "kind": "symbolic", "name": "$H" },
         { "kind": "symbolic", "name": "$S" },
         { "kind": "symbolic", "name": "$D_out" }
       ]
     ],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "symbolic", "name": "$H" },
       { "kind": "symbolic", "name": "$L" },
       { "kind": "symbolic", "name": "$D_out" }
     ]
   }

Symbolic unification ensures that :math:`$B`, :math:`$H`, and :math:`$D`
are consistent across Q, K, and V. The output preserves the query length
:math:`L` and :math:`D_{\text{out}}` from V.

Editor Integration (Phase 5)
----------------------------

The type engine is wired into the visual editor for real-time feedback.

State Management
~~~~~~~~~~~~~~~~

``Diagram.svelte.ts`` holds the inference result as reactive state:

.. code-block:: typescript

   export class Diagram extends DiagramCore {
     public typeResult: TypeResult | null = $state.raw(null);
   }

The ``checkTypes()`` method runs inference on demand:

.. code-block:: typescript

   public checkTypes(): void {
     this.typeResult = TypeEngine.infer(this);
   }

Trigger Events
~~~~~~~~~~~~~~

``checkTypes()`` is called:

* From ``FlowCanvas.svelte`` on edge connect/disconnect and diagram load
* From ``Sidebar.svelte`` on parameter change (debounced ~300ms)

.. code-block:: text

   Edge added ──→ checkTypes() ──→ TypeEngine.infer(diagram)
                                                      │
   Param changed ──→ debounce 300ms ──→ checkTypes() ─┤
                                                      │
   Diagram loaded ──→ checkTypes() ───────────────────┘
                                                      │
                                                      ▼
                                              typeResult updated
                                              ($state.raw reactivity)

Error Display — Sidebar Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A collapsible section at the bottom of the Sidebar lists type errors:

.. code-block:: text

   ┌─ Type Errors (2) ────────────────────────┐
   │ ✗ Linear_1: dimension mismatch:          │
   │   param in_features=512, got 784         │
   │                                          │
   │ ⚠ Fork_0: No type signature for "Fork"  │
   └──────────────────────────────────────────┘

* **Errors** (red background): hard shape mismatches that would cause runtime
  crashes
* **Warnings** (yellow/amber): missing type signatures, unresolved parameters

Clicking an error selects the offending node on the canvas.

Node Error Indicators
~~~~~~~~~~~~~~~~~~~~~

Nodes with type errors display:

* A **red border** (2px) with an ✗ indicator in the top-right corner for errors
* An **amber border** (2px) with a ⚠ indicator for warnings
* Indicators disappear automatically when the error is resolved

Shape Tooltips
~~~~~~~~~~~~~~

Hovering over a node's output handle shows the inferred shape:

.. code-block:: text

   Output: [B, 256]  float32

Phase Summary
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 35 30

   * - Phase
     - Features
     - Modules Covered
     - Implementation
   * - 1
     - Type model, basic pattern matching, data-driven engine
     - Input, Linear, ReLU
     - ``tensortypes.ts``, ``typeEngine.ts``, 3 stereotype JSONs
   * - 2
     - Computed dimensions, shape-preserving modules
     - Conv2d, MaxPool2d, AvgPool2d, Flatten, Tanh, Sigmoid, Softmax, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm, Embedding
     - ``computed`` dim kind, formula resolver, 12 stereotype JSONs
   * - 3
     - Join type checking, multi-input pattern matching, symbolic unification
     - Addition, Concat, MatMul, ScaledDotProduct, MaskedScaledDotProduct
     - Join mode in ``inferNode``, concat constraint, captured dim comparison
   * - 4
     - (Not yet implemented) Subflow type inference, complex modules
     - Subflow, Repeat, HorizontalRepeat, MultiheadAttention, Transformer
     - Future work
   * - 5
     - Editor integration — real-time type feedback
     - All modules with type annotations
     - ``checkTypes()``, error panel, node indicators, shape tooltips

Testing
-------

The type engine has **50+ unit tests** in
``front-end/src/__tests__/typeEngine.test.ts``, organized into groups:

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - Group
     - Tests
     - What They Verify
   * - Happy Path
     - ``Input → Linear``, ``Input → ReLU``, ``Input → Linear → ReLU``, dtype propagation
     - Correct inference on sequential chains
   * - Shape Mismatch
     - ``in_features`` mismatch, chain mismatch (``784→200→?→100``)
     - Errors are detected with correct nodeId and human-readable messages
   * - Edge Cases
     - No type signature (Fork → warning), disconnected node (skipped), empty diagram, join (Einsum → warning)
     - Graceful handling of missing/incomplete type info
   * - Wildcard
     - ``[*]`` captures all dims, preserves through shape-preserving modules
     - Correct wildcard capture and substitution
   * - Computed Dims
     - Formula resolution (conv2d_hw, pool2d_hw, flatten_prod), Conv2d shape inference, Embedding ``[B,L]→[B,L,d]``, shape-preserving chain (Linear → Tanh → Sigmoid → BatchNorm1d)
     - Correct computed dimension resolution
   * - Join Inference
     - Addition (matching + mismatch), Concat on dim=-1 and dim=1, MatMul mismatch, ScaledDotProduct
     - Multi-input pattern matching and symbolic unification

Test helpers in ``front-end/src/__tests__/helpers.ts``:

.. code-block:: typescript

   expectTypeSuccess(result: TypeResult): void
   expectOutputShape(result: TypeResult, nodeId: string, expected: string[]): void
   expectTypeError(result: TypeResult, nodeId: string, messageContains?: string): void

Implementation Details
----------------------

All type system source files are in ``front-end/src/conversion/``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Purpose
   * - ``tensortypes.ts``
     - Type model interfaces (ShapeDimension, TensorType, TypeSignature, TypeResult, ParamResolution)
   * - ``typeEngine.ts``
     - Constraint-based inference engine (TypeEngine.infer, patternMatch, resolvePattern, resolveFormula)
   * - ``Diagram.svelte.ts``
     - Reactive wrapper: ``typeResult`` state, ``checkTypes()`` method

The type system is designed as a **pure, data-driven verification pass** over
the diagram graph. It enriches the DSL with compile-time safety guarantees
while preserving the existing architecture — no changes to NNTree, the Python
backend, or the core diagram state management were required.
