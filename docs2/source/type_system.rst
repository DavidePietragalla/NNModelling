Tensor Type System
==================

A fundamental challenge in visual DSLs for neural network design is the lack
of compile-time feedback about tensor shape compatibility. In NNModelling, a
user may connect a ``Linear`` layer expecting 784 input features to a
``Conv2d`` layer producing a 3D tensor without any editor-level warning. The
shape mismatch is only discovered at runtime when PyTorch raises a dimension
error during training.

To address this, NNModelling includes a **static tensor type system** that
verifies tensor shapes and dtypes during visual editing and front-end
compilation. The type system is integrated into the DSL as a separate
verification pass, preserving the existing architecture while extending the
language with compile-time safety guarantees. It is **data-driven**: all
module-specific logic is declared in ``type_signature`` fields inside
stereotype JSON files, so adding a new module requires only a JSON change —
never a TypeScript modification.

How the Type System Helps
-------------------------

Consider this diagram:

.. code-block:: text

   Input ── Linear(784 → 256) ── ReLU ── Linear(300 → 10) ── CrossEntropyLoss

The second Linear expects 300 input features, but the ReLU outputs 256. This
is a shape mismatch that would cause a runtime crash. Without a type system,
you would only discover this after clicking Convert, training, and reading a
PyTorch error trace.

With the type system, the error appears the moment you connect the second
Linear:

.. code-block:: text

   ✗ Linear_2: dimension mismatch: param in_features=300, got 256

The node turns red. You fix the parameter immediately and move on.

Following a Tensor Through the Graph
-------------------------------------

Let us walk through a well-typed diagram to see how the engine processes
shapes:

.. code-block:: text

   Input(out_features=784) ── Linear(784→256) ── ReLU ── Linear(256→10) ── Loss

**Step 1 — Input**. The Input node is a source: it has no predecessor. Its
type signature declares an output shape :math:`[B, \text{out\_features}]`
where :math:`B` is a fresh symbolic variable representing the batch dimension
and ``out_features`` is read from the node's parameter. With
``out_features=784``, the output type is:

.. code-block:: text

   Tensor([B, 784], float32)

**Step 2 — Linear(784→256)**. The Linear node declares its contract as
:math:`[B, *, \text{in\_features}] \rightarrow [B, *, \text{out\_features}]`.
The wildcard :math:`*` matches zero or more intermediate dimensions. The
engine checks that the incoming last dimension equals the node's
``in_features`` parameter (784 = 784), then produces:

.. code-block:: text

   Tensor([B, 256], float32)

**Step 3 — ReLU**. ReLU is shape-preserving: its signature is
:math:`[*] \rightarrow [*]`. The wildcard captures the entire input shape
:math:`[B, 256]` and reproduces it unchanged in the output.

**Step 4 — Linear(256→10)**. The engine checks that :math:`256 = 256`
(``in_features`` matches), producing:

.. code-block:: text

   Tensor([B, 10], float32)

The entire graph is well-typed: no errors.

Now change the first Linear's ``in_features`` to 512:

.. code-block:: text

   Input ── Linear(512→256) ── ReLU ── Linear(256→10) ── Loss

The Input still produces :math:`[B, 784]`. The first Linear expects its last
dimension to be 512, but the incoming shape is :math:`[B, 784]`. The engine
reports a type error at the Linear node:

.. code-block:: text

   ✗ dimension mismatch: param in_features=512, got 784

The error surfaces immediately in the editor's error panel, and the node
acquires a red border.

Formal Definition
-----------------

We now define the type system formally. The mathematical presentation
follows the standard notation of type theory and is drawn from the project's
academic report.

Tensor Types
~~~~~~~~~~~~

A tensor type :math:`\tau` is a pair consisting of a shape and a data type:

.. math::

   \tau ::= \text{Tensor}(\sigma, \delta)

where :math:`\sigma` is a tensor shape and :math:`\delta` is a tensor data
type (e.g. ``float32``, ``float64``, ``int64``).

Shape Dimensions
~~~~~~~~~~~~~~~~

A shape :math:`\sigma` is a finite sequence of dimensions
:math:`d_1, d_2, \ldots, d_n`. Each dimension :math:`d` belongs to one of
the following categories:

.. math::

   d ::= c \mid x \mid p \mid *

where:

* :math:`c \in \mathbb{N}` is a **constant** dimension (e.g. 3, 784, 1)
* :math:`x \in \mathcal{X}` is a **symbolic** dimension variable (e.g.
  :math:`B` for batch size, :math:`H` for height, :math:`W` for width) —
  these represent unknown dimensions whose values are determined during
  type inference but remain symbolic in the type representation
* :math:`p \in \mathcal{P}` is a **parameter reference** (e.g.
  ``in_features``, ``out_channels``) — these refer to node parameter values
  that are resolved at inference time
* :math:`*` is the **wildcard** dimension, matching zero or more arbitrary
  dimensions. A wildcard in an input pattern consumes matching dimensions
  from the actual tensor; in an output pattern, it reproduces the dimensions
  consumed during input matching

This representation allows the type system to express partially known shapes
— containing symbolic variables and wildcards — rather than requiring every
dimension to be a concrete integer. This is essential for modelling neural
network architectures, where the batch size is unknown at definition time
and intermediate feature dimensions depend on upstream layers.

Typing Context
~~~~~~~~~~~~~~

The typing context :math:`\Gamma` is a partial mapping from symbolic
dimension names to their resolved values:

.. math::

   \Gamma ::= \{ x_1 \mapsto d_1,\; x_2 \mapsto d_2,\; \ldots \}

The context is populated incrementally during type inference as symbolic
dimensions are bound to concrete values. In addition, :math:`\Gamma` carries
dtype information and maintains a mapping from node identifiers to their
inferred tensor types.

Typing Judgments
~~~~~~~~~~~~~~~~

The central judgment form for node-level type inference is:

.. math::

   \Gamma, P \vdash M : (\tau_{\text{in}} \rightarrow \tau_{\text{out}})

meaning: "in context :math:`\Gamma` with parameter values :math:`P`,
module :math:`M` maps input type :math:`\tau_{\text{in}}` to output type
:math:`\tau_{\text{out}}`."

For graph-level inference, the judgment extends to:

.. math::

   \Gamma \vdash G : \Gamma'

meaning: "graph :math:`G` is well-typed, producing the extended environment
:math:`\Gamma'` containing type annotations for every node."

Inference Rules
~~~~~~~~~~~~~~~

The typing rules are defined per module type. Each rule is derived
declaratively from the ``type_signature`` field in the module's stereotype
JSON, rather than being hardcoded in the inference engine. This data-driven
approach ensures that adding a new module requires only extending its
stereotype definition, with no changes to the TypeScript implementation.

**Input Node.** The Input node is a source in the computation graph. It
produces a tensor whose last dimension is determined by its
``out_features`` parameter and whose batch dimension is a fresh symbolic
variable:

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

**Linear Layer.** A Linear layer applies an affine transformation to the
last dimension of its input:

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

where :math:`\alpha_1, \ldots, \alpha_k` are intermediate dimensions matched
by a wildcard pattern and carried forward unchanged. The last dimension
:math:`F` must equal the declared ``in_features`` parameter; if this
constraint is violated, a type error is emitted.

**ReLU Activation (Shape-Preserving).** Activation functions are
shape-preserving and dtype-preserving:

.. math::

   \frac{
     \Gamma \vdash x : \text{Tensor}(\sigma, \delta)
   }{
     \Gamma \vdash \text{ReLU}(x) : \text{Tensor}(\sigma, \delta)
   }

The same rule applies to all shape-preserving modules: Tanh, Sigmoid,
Softmax, Dropout, BatchNorm1d, BatchNorm2d, LayerNorm.

How Modules Declare Their Contracts
------------------------------------

Every stereotype JSON can include a ``type_signature`` field that declares
the module's shape contract. This is the bridge between the formal type
system above and the concrete implementation.

Input
~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [],
     "output": [
       { "kind": "symbolic", "name": "$B" },
       { "kind": "param_ref", "name": "out_features" }
     ],
     "dtype": { "output": "float32" }
   }

The empty ``input`` array means "I have no input — I am a source." The
``output`` says "I produce :math:`[B, \text{out\_features}]` with dtype
``float32``."

Linear
~~~~~~

.. code-block:: json

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

The pattern :math:`[B, *, \text{in\_features}]` means: match a batch
dimension, then zero or more intermediate dimensions captured by the
wildcard, then require the last dimension to equal the ``in_features``
parameter. The output preserves the batch and intermediate dimensions while
replacing the last dimension with ``out_features``.

A naming convention applies to all JSON type signatures: symbolic names
start with ``$`` (``"$B"``, ``"$H"``, ``"$W"``) to distinguish them from
parameter references, which never have the prefix. The ``$`` is stripped
when the JSON is loaded into the engine.

ReLU and Shape-Preserving Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [{ "kind": "wildcard" }],
     "output": [{ "kind": "wildcard" }]
   }

The wildcard on both input and output means "I accept any shape, and the
output shape equals the input shape." This single pattern covers ReLU, Tanh,
Sigmoid, Softmax, Dropout, BatchNorm1d, BatchNorm2d, and LayerNorm.

The Notation Rule for ``$``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In JSON, symbolic dimension names start with ``$`` (``"$B"``, ``"$H"``,
``"$W"``). Parameter references never have ``$`` (``"in_features"``,
``"out_channels"``). When loaded, ``"$B"`` becomes ``{ kind: 'symbolic',
name: 'B' }`` — the ``$`` is stripped, leaving only the canonical name.

The Inference Engine
--------------------

The type engine (``TypeEngine`` in ``typeEngine.ts``) implements the formal
rules above as a constraint-based algorithm operating in two phases:

**1. Constraint Generation.** For each node visited in topological order,
the engine reads the node's ``type_signature`` and generates constraints by
pattern-matching the actual input shape against the declared input pattern.
This produces:

* **Bindings**: symbolic dimensions in the pattern are bound to their
  matched concrete values
* **Substitutions**: wildcard dimensions capture a suffix of the actual
  shape for reuse in the output pattern
* **Resolutions**: parameter references are resolved to the node's current
  parameter values

**2. Constraint Solving.** The engine substitutes bound variables and
captured wildcards into the output pattern, producing the output tensor
type. If any constraint is violated (a constant dimension does not match,
a dtype constraint fails, a parameter reference cannot be resolved), a
``TypeError`` is recorded.

The algorithm in pseudocode:

.. code-block:: text

   TypeEngine.infer(diagram):
     1. Build topological order (Kahn's algorithm on top-level nodes)
     2. For each node in order:
        a. Read stereotype and typeSignature
        b. If no typeSignature → warning, treat output as "unknown"
        c. Determine input type(s) from predecessor annotations
        d. Call patternMatch(inputShape, inputPattern, params, env)
        e. If match fails → record TypeError, continue
        f. Resolve output: substitute bindings + captured wildcards
        g. Store annotation, update environment
     3. Return { ok, annotations, errors }

Pattern Matching
~~~~~~~~~~~~~~~~

The core pattern matching algorithm walks input dimensions and pattern
elements with a two-pointer approach:

+-------------+----------------------------------------------------------+
| Kind        | Behaviour                                                |
+=============+==========================================================+
| ``const``   | The input dimension at this position must equal the      |
|             | constant value. E.g.: pattern expects 3, input has 1 →   |
|             | ``dimension mismatch: expected 3, got 1``                |
+-------------+----------------------------------------------------------+
| ``symbolic``| If the name is already bound in :math:`\Gamma`, the      |
|             | input dimension must equal the bound value (unification). |
|             | If unbound, a new binding is added. E.g.: ``symbolic $B  |
|             | already bound to 784, cannot unify with 128``            |
+-------------+----------------------------------------------------------+
|``param_ref``| The parameter is resolved from the node's parameter map. |
|             | The input dimension must equal the resolved value. If    |
|             | the value is non-numeric (e.g. "cazz"), a type error is  |
|             | reported rather than silently treating it as unset.      |
+-------------+----------------------------------------------------------+
| ``wildcard``| Consumes zero or more input dimensions, with lookahead   |
|             | for subsequent required pattern elements. The consumed   |
|             | dimensions are captured and substituted into the output.  |
+-------------+----------------------------------------------------------+
| ``computed``| Pass-through on the input side (exact value is not       |
|             | validated); the formula is resolved on the output side.  |
+-------------+----------------------------------------------------------+

**Wildcard lookahead.** The wildcard does not blindly consume all remaining
dimensions. It computes how many are needed for subsequent non-wildcard
pattern elements and reserves them. For the pattern :math:`[B, *,
\text{in\_features}]` on input :math:`[B, 128, 784]`: the wildcard consumes
one dimension (128), leaving the last for ``in_features``. On input
:math:`[B, 784]` it consumes zero dimensions.

**Symbolic unification.** When a symbolic name appears in multiple positions
(e.g. :math:`\$K` in both MatMul input patterns), the engine verifies that
all occurrences bind to the same concrete value. If they conflict, a
unification error is reported.

Computed Dimensions
-------------------

Some modules produce output dimensions that depend on their parameters in
non-trivial ways. These are expressed through the ``computed`` dimension
kind with named formulas.

Conv2d
~~~~~~

The output height and width of a convolution are computed from the input
size, kernel size, stride, padding, and dilation:

.. math::

   H_{\text{out}} = \left\lfloor \frac{H + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor

The type signature declares this as:

.. code-block:: json

   "output": [
     { "kind": "symbolic", "name": "$B" },
     { "kind": "param_ref", "name": "out_channels" },
     { "kind": "computed", "formula": "conv2d_hw",
       "args": ["$H", "kernel_size", "stride", "padding", "dilation"] },
     { "kind": "computed", "formula": "conv2d_hw",
       "args": ["$W", "kernel_size", "stride", "padding", "dilation"] }
   ]

For a 3×3 convolution with stride 1 and padding 1 on a 32×32 input:
:math:`(32 + 2(1) - 1(3-1) - 1) / 1 + 1 = 32` (the output is the same size).

MaxPool2d and AvgPool2d
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} + 1 \right\rfloor

For 2×2 pooling with stride 2 on a 32×32 input:
:math:`(32 + 0 - 2) / 2 + 1 = 16` (halves the spatial dimensions).

Flatten
~~~~~~~

The flattened dimension is the product of all wildcard-captured dimensions
(referenced as ``$*`` in the formula arguments):

.. math::

   d_{\text{flat}} = \prod_i d_i

For :math:`[B, 128, 7, 7]`: :math:`128 \times 7 \times 7 = 6272` →
:math:`[B, 6272]`.

Join Type Checking
------------------

Join nodes accept multiple inputs and merge them into one. The type engine
validates multi-input shape compatibility through pattern matching and
symbolic unification.

**Addition** requires all inputs to have identical shapes. The engine
captures dimensions from the first input's wildcard and verifies that
subsequent inputs produce identical captured dimensions. If one branch
produces :math:`[B, 256]` and another produces :math:`[B, 128]`:

.. code-block:: text

   ✗ Addition: Input 1 dimension 1 mismatch: 256 vs 128

**Concat** concatenates along a specified dimension (:math:`d`). All other
dimensions must match. The output on dimension :math:`d` is the sum of the
input dimensions:

.. math::

   \text{Concat}(\text{dim}=d)(x_1, \ldots, x_n)[d] = \sum_{i=1}^n x_i[d]

For :math:`[B, 128]` and :math:`[B, 64]` with ``dim=-1``, the output is
:math:`[B, 192]`.

**MatMul** constrains the inner dimensions to match through symbolic
unification:

.. math::

   (M, K) \times (K, N) \rightarrow (M, N)

If the first input is :math:`(32, 64)` and the second is
:math:`(128, 64)`, the engine binds :math:`K = 64` from the first and
cannot unify with 128 from the second.

**ScaledDotProduct** validates the full attention shape pattern:

.. math::

   Q(B, H, L, D) \times K(B, H, S, D) \times V(B, H, S, D_{\text{out}}) \rightarrow (B, H, L, D_{\text{out}})

Symbolic unification ensures that :math:`B`, :math:`H`, and :math:`D` are
consistent across Q, K, and V. The output preserves the query length
:math:`L` and value depth :math:`D_{\text{out}}` from V.

***Note.** Einsum has no type signature because its shape constraints depend
on the equation string, which is a free-form parameter. The engine emits a
warning and treats its type as unknown. This is a deliberate case of
gradual typing.*

Real-Time Feedback in the Editor
---------------------------------

The type engine is wired into the visual editor so that errors surface
immediately as the user edits.

**Trigger events.** ``TypeEngine.infer(diagram)`` is called every time an
edge is added or removed, a parameter changes (debounced 300ms), or a
diagram is loaded.

**Error panel.** A collapsible section at the bottom of the Sidebar lists
all type problems. Errors (red) indicate shape mismatches that would cause
runtime crashes. Warnings (amber) indicate missing type signatures or
unresolved parameters. Clicking an error selects the offending node.

**Node indicators.** Nodes with errors display a red border (2px) with an
✗ indicator in the top-right corner. Nodes with warnings display an amber
border with a ⚠ indicator. Indicators disappear when the error is resolved.

**Shape tooltips.** Hovering over a node's output handle shows a tooltip
with the inferred output shape:

.. code-block:: text

   Output: [B, 256]  float32

Implementation Phases
---------------------

The type system was implemented incrementally over five phases:

+-------------+----------------------------------------------------------+
| Phase       | What Was Added                                           |
+=============+==========================================================+
| Phase 1     | Core type model (tensortypes.ts), pattern matching        |
|             | engine, type signatures for Input/Linear/ReLU            |
+-------------+----------------------------------------------------------+
| Phase 2     | Computed dimensions (conv2d_hw, pool2d_hw,               |
|             | flatten_prod), Conv2d/MaxPool2d/AvgPool2d/Flatten,      |
|             | 10 shape-preserving modules, Embedding                   |
+-------------+----------------------------------------------------------+
| Phase 3     | Join type checking (Addition, Concat, MatMul,            |
|             | ScaledDotProduct, MaskedScaledDotProduct),               |
|             | multi-input pattern matching, symbolic unification       |
+-------------+----------------------------------------------------------+
| Phase 4     | Recursive subflow type inference (generic, Repeat,        |
|             | HorizontalRepeat), type signatures for 7 complex modules |
|             | (MultiheadAttention, Transformer, TransformerEncoderLayer,|
|             | TransformerDecoderLayer, PositionalEncoding, SequencePool,|
|             | Unsample), 4 loss nodes (BCELoss, BCEWithLogitsLoss,     |
|             | CrossEntropyLoss, MSELoss), Fork, and the ``upsample_hw`` |
|             | computed formula. 34 of 35 stereotypes now have type      |
|             | signatures (Einsum is intentionally gradual typing).      |
+-------------+----------------------------------------------------------+
| Phase 5     | Editor integration: reactive ``typeResult`` state,       |
|             | error panel, node indicators, shape tooltips             |
+-------------+----------------------------------------------------------+

Further Reading
---------------

* :doc:`stereotypes` — how modules declare their JSON type signatures
* Source: ``front-end/src/conversion/tensortypes.ts`` — type model interfaces
* Source: ``front-end/src/conversion/typeEngine.ts`` — inference engine
  implementation
* Tests: ``front-end/src/__tests__/typeEngine.test.ts`` — 170 tests
  (165 passing, 5 skipped) covering all phases
* Design docs: ``docs/designs/tensor-type-system/`` — full architectural
  design documents
