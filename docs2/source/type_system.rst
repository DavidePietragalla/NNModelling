Understanding Tensor Types
==========================

Have you ever connected a ``Linear(784 → 256)`` to a ``Conv2d`` in the visual
editor, clicked **Convert**, trained for an hour, and then seen this?

.. code-block:: text

   RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 512x256)

You stare at the error. You open your diagram. You trace connections manually.
Where is the mismatch? It takes five minutes to find the node where you typed
``512`` instead of ``784``. You fix it, reconvert, retrain. Five minutes of
your life, gone.

The **tensor type system** is here to catch that mistake the moment you make
it — while you are still editing the diagram, before you even click Convert.

.. code-block:: text

   ┌─ Type Errors (1) ────────────────────────┐
   │ ✗ Linear_1: dimension mismatch:          │
   │   param in_features=512, got 784         │
   └──────────────────────────────────────────┘

The node turns red. You see the error immediately. You fix it, save, move on.
No wasted training runs.

What Problem Does It Solve?
---------------------------

In a visual neural network editor, every node is a black box that transforms
a tensor. The editor knows what parameters the user typed (e.g.
``in_features=512``), but without a type system it cannot check whether those
parameters make sense in context.

Consider this diagram:

.. code-block:: text

   Input ── Linear(784 → 256) ── ReLU ── Linear(300 → 10) ── CrossEntropyLoss

The second Linear expects ``in_features=300``. But the ReLU outputs 256
features. This is a shape mismatch — PyTorch will crash at runtime.

The type system detects this automatically. It follows the tensor shape
through every node and flags the mismatch the moment you connect the second
Linear.

Following a Tensor Through the Graph
-------------------------------------

Let us walk through a simple example to see how the type system thinks.

We have this diagram:

.. code-block:: text

   Input(out_features=784) ── Linear(784→256) ── ReLU ── Linear(256→10) ── Loss

**Step 1: Input**. The Input node says: "I produce a tensor with shape
[B, 784] and dtype float32." The B is a symbolic batch dimension — we do
not know its value yet, but we know it exists.

   ``output: [B, 784] (float32)``

**Step 2: Linear(784→256)**. The Linear node says: "I expect input shape
[B, *, in_features=784]. The wildcard * means I can handle zero or more
intermediate dimensions." It looks at the incoming [B, 784], finds that the
last dimension 784 matches its ``in_features`` parameter, and produces
[B, 256]:

   ``output: [B, 256] (float32)``

**Step 3: ReLU**. The ReLU says: "I don't change shapes. Whatever comes
in, I pass through." It outputs the same [B, 256].

**Step 4: Linear(256→10)**. It expects [B, *, in_features=256]. The incoming
shape is [B, 256]. Last dimension 256 matches ``in_features=256``. Output:
[B, 10].

Everything works. The type system reports no errors.

**Now let us introduce an error.** Change the first Linear to
``in_features=512``:

.. code-block:: text

   Input ── Linear(512→256) ── ReLU ── Linear(256→10) ── Loss

The Input still produces [B, 784]. The Linear says "I expect
[B, *, in_features=512]". It looks at the incoming last dimension: it is 784,
not 512. Mismatch.

   ``✗ Linear_1: dimension mismatch: param in_features=512, got 784``

The node turns red. The error panel shows the message. You fix it instantly.

The Language of Shapes
----------------------

Every module declares its tensor shape contract using a small language of
**dimension kinds**. There are five of them:

``const`` — a fixed number
   Dimensions like 784, 3, 64, 10. When a module expects a ``const`` dimension,
   the incoming shape must match exactly.

   .. code-block:: json

      { "kind": "const", "value": 784 }

``symbolic`` — a named variable
   Symbolic dimensions connect matching values across a module. When a module
   uses the same symbolic name in both input and output (e.g. ``$B`` for batch
   size), it means "whatever value this dimension has on the input, keep it
   unchanged on the output."

   In JSON, symbolic names are written with a ``$`` prefix (``"$B"``, ``"$H"``,
   ``"$W"``) to distinguish them from parameter references. The prefix is
   stripped when loaded into the engine.

   .. code-block:: json

      { "kind": "symbolic", "name": "$B" }

``param_ref`` — a value from the node's parameters
   When a module refers to a parameter like ``in_features`` or ``out_channels``,
   it means "read the value from this node's configuration." If the user typed
   ``512`` for ``in_features``, the engine checks that the incoming dimension
   equals 512.

   .. code-block:: json

      { "kind": "param_ref", "name": "in_features" }

``wildcard`` — "I do not care about these dimensions"
   A wildcard matches zero or more arbitrary dimensions. It captures them from
   the input and reproduces them at the corresponding position in the output.
   This is how modules like Linear can preserve intermediate dimensions while
   transforming only the last one.

   The pattern ``[B, *, in_features]`` means: match a batch dim, then skip
   any number of intermediate dims, then match the last dim against
   ``in_features``. If the input is ``[B, 128, 784]``, the wildcard captures
   ``128`` and passes it through to the output. If the input is ``[B, 784]``,
   the wildcard captures nothing.

   .. code-block:: json

      { "kind": "wildcard" }

``computed`` — a dimension calculated by formula
   Some modules (like Conv2d) produce dimensions that depend on parameters and
   input dimensions in non-trivial ways. The output height of a Conv2d depends
   on the input height, kernel size, stride, padding, and dilation.

   Computed dimensions use named formulas (``conv2d_hw``, ``pool2d_hw``,
   ``flatten_prod``) that are resolved when all arguments are known.

   .. code-block:: json

      { "kind": "computed", "formula": "conv2d_hw",
        "args": ["$H", "kernel_size", "stride", "padding", "dilation"] }

The ``$`` prefix rule
~~~~~~~~~~~~~~~~~~~~~

In stereotype JSON, symbolic names start with ``$`` (``"$B"``, ``"$H"``).
Parameter references never have ``$`` (``"in_features"``, ``"out_channels"``).
This distinction makes the JSON self-documenting — you can see at a glance
which names are variables and which are parameter lookups.

The ``$`` is stripped when the JSON is loaded into the engine: ``"$B"`` becomes
``{ kind: 'symbolic', name: 'B' }``.

How Modules Describe Their Shape Contracts
------------------------------------------

Every module declares its expected input shape and produced output shape in its
stereotype JSON, in a field called ``type_signature``.

Think of the type signature as the module's **contract**: "If you give me a
tensor that matches this input pattern, I will give you a tensor matching this
output pattern."

Input — the entry point
~~~~~~~~~~~~~~~~~~~~~~~

The Input node has no predecessors. It produces a tensor with a symbolic batch
dimension and a feature dimension controlled by its parameter:

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

The empty ``input`` array means "I have no input — I am a source."
The ``output`` says "I produce [B, out_features] with dtype float32."

Linear — transforming the last dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear applies an affine transformation to the last dimension of its input.
It keeps the batch dimension and any intermediate dimensions:

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

This pattern says: "I need a batch dim, then I do not care how many
intermediate dimensions there are, and the last dimension must equal
my ``in_features``. I preserve the batch, preserve the intermediate dims,
and change the last dimension to ``out_features``."

ReLU and friends — shape-preserving modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activation functions, dropout, and normalization layers do not change shapes.
Their contract is simply: "whatever comes in goes out unchanged":

.. code-block:: json

   "type_signature": {
     "kind": "module",
     "input": [{ "kind": "wildcard" }],
     "output": [{ "kind": "wildcard" }]
   }

The wildcard on both input and output means "I accept any shape, and the
output has the same shape as the input."

This single pattern covers: ReLU, Tanh, Sigmoid, Softmax, Dropout,
BatchNorm1d, BatchNorm2d, LayerNorm.

Conv2d — computed output dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolutional layers change the spatial dimensions of their input. The output
height and width are computed from the input dimensions and parameters:

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

The ``computed`` dimensions use the formula:

.. math::

   H_{\text{out}} = \left\lfloor \frac{H + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor

The engine resolves this formula when all arguments (``$H``, ``kernel_size``,
``stride``, etc.) are known. If any argument is still symbolic (e.g. the input
height is not yet known), the dimension remains as a deferred computation.

Embedding — adding a dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Embedding takes token indices and produces dense vectors:

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

The input is ``[B, L]`` (batch and sequence length). The output adds a third
dimension: ``[B, L, embedding_dim]``.

When Two Branches Meet: Join Nodes
-----------------------------------

Joins are nodes that accept multiple inputs and merge them into one. The type
system must check that all incoming branches produce compatible shapes.

Addition — everything must match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Element-wise addition requires all inputs to have identical shapes:

.. code-block:: json

   "type_signature": {
     "kind": "join",
     "input": [
       [{ "kind": "wildcard" }],
       [{ "kind": "wildcard" }]
     ],
     "output": [{ "kind": "wildcard" }]
   }

The engine captures the shape from the first input and checks that every
subsequent input has the same captured dimensions. If one branch produces
[B, 256] and another produces [B, 128]:

.. code-block:: text

   ✗ Addition: Input 1 dimension 1 mismatch: 256 vs 128

Concat — summing on one axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenation joins tensors along a specified dimension. All other dimensions
must match:

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

The ``concat`` constraint says "the output dimension on this axis is the sum
of the input dimensions." For two inputs [B, 128] and [B, 64] with
``dim=-1``:

.. code-block:: text

   Input 0: [B, 128]
   Input 1: [B,  64]
   ─────────────────
   Output:  [B, 192]

MatMul — connecting two dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matrix multiplication has a precise shape constraint:

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

The symbolic variable ``$K`` appears in both inputs — this means the engine
**unifies** them: whatever value ``$K`` has in the first input must match its
value in the second. If the first input is (M, 64) and the second is (128, N),
the engine binds K=64 from the first and then detects that 64 ≠ 128 in the
second, reporting:

.. code-block:: text

   ✗ MatMul: symbolic $K already bound to 64, cannot unify with 128

ScaledDotProduct — attention shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Attention requires careful shape coordination across three inputs:

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

The engine ensures that:
* Batch (``$B``) and head count (``$H``) are consistent across Q, K, V
* The key/value sequence lengths (``$S``) match between K and V
* The query/key depth (``$D``) matches between Q and K
* The output preserves the query length (``$L``) and value depth (``$D_out``)

How the Engine Works
--------------------

The type engine is a **pure, data-driven inference pass** over the diagram
graph. It does not know about Svelte, the user interface, or the Python
backend. It simply looks at nodes, edges, and stereotype JSONs, and produces
type annotations.

The algorithm is straightforward:

1. **Sort the nodes in execution order** (topological sort). Start with nodes
   that have no incoming edges (the Input node), then follow connections
   forward.

2. **Start at the Input node.** Its output shape is defined by its
   ``type_signature`` and parameters.

3. **For each subsequent node, check the contract.** Look at the incoming
   tensor shape. Compare it against the node's declared input pattern. If
   they match, compute the output shape from the output pattern. If they
   do not match, record a type error.

4. **Pass the output shape forward.** The next node receives this as its
   input shape, and the process repeats.

If a cycle is detected (rare — the editor prevents most cycles), the engine
emits a warning and processes only the nodes reachable from the Input.

What about modules that have no ``type_signature``? The engine emits a warning
and treats their output as "unknown type." This is called **gradual typing** —
modules without signatures still work, they just do not get checked.

Here is what the full inference looks like for our example:

.. code-block:: text

   Input(out_features=784)
     │
     ▼ output: [B, 784] (float32)
     │
   Linear(in_features=784, out_features=256)
     │ patternMatch: [B, 784] vs [B, *, in_features=784] → OK
     │ output: [B, 256] (float32)
     │
     ▼
   ReLU
     │ patternMatch: [B, 256] vs [*] → captured [B, 256]
     │ output: [B, 256] (float32)
     │
     ▼
   Linear(in_features=300, out_features=10)
     │ patternMatch: [B, 256] vs [B, *, in_features=300] → FAIL
     │ "param in_features=300, got 256"
     │
     ▼
   ✗ ERROR — node marked with red border

Three core ideas make this engine work:

**Data-driven design**: The engine never checks ``if (stereotype.name === 'Linear')``.
It reads the ``type_signature`` from the JSON and follows whatever patterns
it finds. Adding a new module requires only adding a ``type_signature``
field to its JSON — never a code change.

**Pattern matching with wildcard lookahead**: When the engine sees a wildcard
in a pattern like ``[B, *, in_features]``, it does not blindly consume all
remaining dimensions. It looks ahead to see what other pattern elements follow,
reserves dimensions for them, and captures only the excess. A pattern like
``[B, *, in_features]`` on input ``[B, 128, 784]`` captures one dimension
(128) because the last dimension is needed for ``in_features``. On input
``[B, 784]`` it captures nothing.

**Parameter validation**: When the engine reads a parameter like
``in_features``, it returns one of three results:

* ``'resolved'`` — the parameter has a valid numeric value (e.g. 784)
* ``'unset'`` — the parameter is ``"None"`` or missing (treated as a soft
  warning, not an error)
* ``'invalid'`` — the parameter has a non-numeric value like ``"cazz"`` or
  ``"hello"`` (reported as a type error)

Computed Dimensions: Conv2d and Flatten
---------------------------------------

Some modules produce output dimensions that depend on their parameters in
non-trivial ways. The type system handles these through **named formulas**.

``conv2d_hw``: used by Conv2d
   The output height and width of a convolution depend on the input size,
   kernel size, stride, padding, and dilation:

   .. math::

      H_{\text{out}} = \left\lfloor \frac{H + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor

   For a 3×3 convolution with stride 1 and padding 1 on a 32×32 input:
   ``(32 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 32`` (the output is the same size).

``pool2d_hw``: used by MaxPool2d, AvgPool2d
   Pooling has a simpler formula:

   .. math::

      H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} + 1 \right\rfloor

   For 2×2 max pooling with stride 2 on a 32×32 input:
   ``(32 + 0 - 2) / 2 + 1 = 16`` (halves the spatial dimensions).

``flatten_prod``: used by Flatten
   The flattened dimension is the product of all feature dimensions:

   .. math::

      d_{\text{flat}} = \prod_i d_i

   For a ``[B, 128, 7, 7]`` tensor: ``128 * 7 * 7 = 6272`` → output ``[B, 6272]``.

These formulas are resolved lazily: if an input dimension is still symbolic
(e.g. you have not connected a source that fixes the width), the computed
dimension remains as a deferred computation. Once all inputs are known, the
formula is evaluated and the dimension becomes a concrete number.

Real-Time Feedback in the Editor
--------------------------------

The type system is wired into the visual editor so you see errors as you work.

When does the engine run?
~~~~~~~~~~~~~~~~~~~~~~~~~

Every time you:

* **Add or remove a connection** — the engine re-checks the affected path
* **Change a parameter** — after a short delay (300ms debounce) to avoid
  checking on every keystroke
* **Load a diagram** — the engine checks the entire graph

What do you see?
~~~~~~~~~~~~~~~~

**An error panel** at the bottom of the sidebar lists all type problems:

.. code-block:: text

   ┌─ Type Errors (2) ────────────────────────┐
   │ ✗ Linear_1: dimension mismatch:          │
   │   param in_features=512, got 784         │
   │                                          │
   │ ⚠ Fork_0: No type signature for "Fork"  │
   └──────────────────────────────────────────┘

* Red items are **errors** — shape mismatches that would crash training
* Amber items are **warnings** — modules without type signatures, unset
  parameters that the engine cannot check

Clicking an error selects the offending node on the canvas, so you can fix
it immediately.

**Colored node borders** let you spot problems at a glance:

* **Red border** with ✗ icon — this node has a type error
* **Amber border** with ⚠ icon — this node has a warning

**Shape tooltips** appear when you hover over a node's output handle:

.. code-block:: text

   Output: [B, 256]  float32

This lets you check what shape a node actually produces before connecting
it to the next layer.

What's Next?
------------

The type system was implemented in phases. Here is where we are and what
is coming:

+-------------+--------------------------------------------------+
| Phase       | What Was Added                                   |
+=============+==================================================+
| Phase 1     | Core type model, basic pattern matching,         |
|             | Input / Linear / ReLU type signatures            |
+-------------+--------------------------------------------------+
| Phase 2     | Computed dimensions (Conv2d, MaxPool2d,          |
|             | Flatten), 10 shape-preserving modules,           |
|             | Embedding                                        |
+-------------+--------------------------------------------------+
| Phase 3     | Join type checking (Addition, Concat, MatMul,    |
|             | ScaledDotProduct, MaskedScaledDotProduct)        |
+-------------+--------------------------------------------------+
| Phase 4     | **(Future)** Subflow type inference (Repeat,     |
|             | HorizontalRepeat, MultiheadAttention)            |
+-------------+--------------------------------------------------+
| Phase 5     | Editor integration: error panel, node borders,   |
|             | shape tooltips, real-time checking               |
+-------------+--------------------------------------------------+

The most impactful near-term addition is Phase 4 — subflow type inference.
Currently, subflows pass through as "unknown type." Once implemented, the
engine will recursively check the internal graph of subflows, making the
type system fully comprehensive.

Further Reading
---------------

* :doc:`stereotypes` — how modules declare their JSON type signatures
* Source: ``front-end/src/conversion/tensortypes.ts`` — type model interfaces
* Source: ``front-end/src/conversion/typeEngine.ts`` — inference engine
  implementation
* Tests: ``front-end/src/__tests__/typeEngine.test.ts`` — 50+ tests covering
  happy paths, mismatches, edge cases, computed dims, and joins
* Design docs: ``docs/designs/tensor-type-system/`` — full architectural
  design documents
