Examples
========

NNModelling ships with 10 pre-built example diagrams in
``examples/diagrams/``. These demonstrate different network architectures,
stereotype combinations, and pipeline features.

Running examples
----------------

You can run any example through the full pipeline using:

.. code-block:: bash

   NNM_DIAGRAM=<name> pnpm run test:example

Where ``<name>`` is the diagram filename without the ``.json`` extension
and without the ``examples/diagrams/`` path prefix.

To run a specific integration tier on an example:

.. code-block:: bash

   NNM_DIAGRAM=mninst NNM_TIER=convert pnpm run test:integration:convert

Each example has a corresponding ``examples/nntrees/`` pre-compiled NNTree
JSON that can be used directly with the Python pipeline:

.. code-block:: bash

   cd converted
   uv run python src/convert.py ../examples/nntrees/mninst_skip.json ./output
   uv run python src/main.py --config-dir ./output

---

Example 1: MNIST Classifier (mninst)
-------------------------------------

**File:** ``examples/diagrams/mninst.json``
**NNTree:** ``examples/nntrees/n/a`` (NNTree not pre-compiled)

A simple multi-layer perceptron (MLP) for MNIST digit classification.

.. code-block:: text

   Input (1x28x28)
     │
     ▼
   Flatten
     │
     ▼
   Linear (784 → 128)
     │
     ▼
   ReLU
     │
     ▼
   Linear (128 → 64)
     │
     ▼
   ReLU
     │
     ▼
   Linear (64 → 10)
     │
     ▼
   CrossEntropyLoss

**What it demonstrates:**

* Basic sequential chain of modules
* Flatten from 2D image to 1D features
* Multi-class classification with CrossEntropyLoss
* The simplest end-to-end pipeline

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=mninst pnpm run test:example

**Expected output:** Training reaches >95% test accuracy within 1-2 epochs.

---

Example 2: MNIST with Skip Connections (mnist_skips)
------------------------------------------------------

**File:** ``examples/diagrams/mnist_skips.json``
**NNTree:** ``examples/nntrees/mninst_skip.json``

An MLP with skip connections using Addition joins.

.. code-block:: text

   Input ─── Linear(784→128) ─── ReLU ──┐
        │                                │
        └── Linear(784→64)  ─── ReLU ─── Addition ─── Linear(64→10) ─── CrossEntropyLoss

**What it demonstrates:**

* Skip connections with Addition join
* Implicit forks (Input connects to two branches)
* The same layer can feed into multiple downstream nodes

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=mnist_skips pnpm run test:example

---

Example 3: Convolutional Autoencoder (autoencoder_mnist)
---------------------------------------------------------

**File:** ``examples/diagrams/autoencoder_mnist.json``

An encoder-decoder autoencoder for MNIST reconstruction.

.. code-block:: text

   Input (1×28×28)
     │
     ▼
   Conv2d(1→16, k=3) ─── ReLU ─── MaxPool2d ─── Conv2d(16→8, k=3) ─── ReLU ─── MaxPool2d
     │
     ▼ [bottleneck: 8×5×5]
     │
   Conv2d(8→16, k=3) ─── ReLU ─── Upsample(×2) ─── Conv2d(16→1, k=3) ─── Sigmoid
     │
     ▼
   MSELoss

**What it demonstrates:**

* Convolutional layers (Conv2d, MaxPool2d)
* Upsampling with Upsample (``mode='bilinear'``)
* MSELoss for reconstruction tasks
* Autoencoder architecture (symmetric encode/decode)
* Sigmoid activation for pixel output (values in [0,1])

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=autoencoder_mnist pnpm run test:example

**Expected output:** Reconstructed images should resemble the input digits
after training. Inference can generate visual comparisons using ``--image-dir``.

---

Example 4: Autoencoder with SubFlows (auto_encoder_submodels)
--------------------------------------------------------------

**File:** ``examples/diagrams/auto_encoder_submodels.json``

Same autoencoder architecture as Example 3, but with the encoder and decoder
wrapped in SubFlow containers.

.. code-block:: text

   Input ─── [Encoder SubFlow] ─── [Decoder SubFlow] ─── MSELoss

**What it demonstrates:**

* SubFlow containers for logical grouping
* Encapsulating encoder/decoder as reusable blocks
* SubFlows preserve internal DAG topology (not flattened)

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=auto_encoder_submodels pnpm run test:example

---

Example 5: Nested SubFlows (auto_encoder_submodels_with_submodels)
-------------------------------------------------------------------

**File:** ``examples/diagrams/auto_encoder_submodels_with_submodels.json``

Autoencoder with subflows inside subflows — demonstrating recursive nesting.

**What it demonstrates:**

* Arbitrarily nested subflow containers
* Recursive ``compileSubflowGraph`` in nnTree.ts
* SubFlow boundary detection at multiple levels

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=auto_encoder_submodels_with_submodels pnpm run test:example

---

Example 6: Single-Head Attention (single_head_attention)
---------------------------------------------------------

**File:** ``examples/diagrams/single_head_attention.json``

A single-head attention mechanism built from primitives.

.. code-block:: text

   Input
     ├── Linear(d→d) ────┐  (Query projection)
     ├── Linear(d→d) ────┤  (Key projection)
     └── Linear(d→d) ────┘  (Value projection)
              │
              ▼
          ScaledDotProduct ─── Linear(d→d) ──── Output

**What it demonstrates:**

* Building attention from scratch with primitive operations
* ScaledDotProduct join (Q, K, V → attention output)
* Forks (Input → 3 Linear projections)
* Projection matrices as Linear layers

---

Example 7: Multi-Head Attention via Concat (multihead_attention)
-----------------------------------------------------------------

**File:** ``examples/diagrams/multihead_attention.json``

4-head attention built by concatenating 4 single-head attention blocks.

.. code-block:: text

   Input
     │
     ├── [Head 1: Linear→ScaledDotProduct→Linear] ──┐
     ├── [Head 2: Linear→ScaledDotProduct→Linear] ──┤
     ├── [Head 3: Linear→ScaledDotProduct→Linear] ──┤
     └── [Head 4: Linear→ScaledDotProduct→Linear] ──┘
              │
         Concat(dim=-1) ──── Linear(4d→d) ──── Output

**What it demonstrates:**

* Explicit multi-head design via Concat join
* Each head is a duplicate of the single-head pattern
* The Concat join combines outputs along the feature dimension
* Linear projection to reduce ``4d`` back to ``d``

---

Example 8: Multi-Head Attention via HorizontalRepeat (horizontal_multihead_attention)
---------------------------------------------------------------------------------------

**File:** ``examples/diagrams/horizontal_multihead_attention.json``

4-head attention using the HorizontalRepeat subflow stereotype.

.. code-block:: text

   Input ─── [HorizontalRepeat n=4: Single-Head Attention SubFlow] ─── Output

**What it demonstrates:**

* HorizontalRepeat subflow for compact multi-head design
* ``n=4`` creates 4 parallel copies via ``vmap``
* The join is hardcoded to concat on ``dim=-1``
* Much more compact than explicit 4-head wiring
* Uses ``functional_call`` with stacked module state

**Advantage over Example 7:** Only one head needs to be designed visually.
The ``HorizontalRepeat`` stereotype automatically creates N copies with
independent weights.

---

Example 9: Residual + Repeat (skip_connections_with_repetition)
-----------------------------------------------------------------

**File:** ``examples/diagrams/skip_connections_with_repetition.json``
**NNTrees:** ``examples/nntrees/skip_connections_with_repetition.json``,
``examples/nntrees/skip_connections_without_repetition.json``

A residual block wrapped in a Repeat subflow.

.. code-block:: text

   Input ─── [Repeat iterations=3: Residual Block] ─── Output

   Residual Block (inside subflow):
     Input ─── Fork ─── Linear(64→64) ─── ReLU ─── Linear(64→64) ───┐
          │                                                         │
          └──────────────────────────────────────────────────────── Addition

**What it demonstrates:**

* Repeat subflow for deep residual networks
* Fork stereotype inside subflow (explicit passthrough node)
* Addition join for skip/residual connection
* Combining Repeat with Fork and Addition for ResNet-style blocks
* The same pattern can scale to 10, 50, or 100 layers by changing
  ``iterations``

---

Example 10: Transformer Classifier (transformer_classifier)
-------------------------------------------------------------

**File:** ``examples/diagrams/transformer_classifier.json``
**NNTree:** ``examples/nntrees/transformer_classifier.json``

A full transformer encoder for text classification.

.. code-block:: text

   Input (token_ids)
     │
     ▼
   Embedding(vocab=10000, dim=256)
     │
     ▼
   PositionalEncoding(d_model=256)
     │
     ▼
   TransformerEncoderLayer(d_model=256, nhead=8) ────┐
     │                                                 │
     ▼                                                 │
   TransformerEncoderLayer(d_model=256, nhead=8)       │ (Repeat iterations=2)
     │                                                 │
     ▼                                                 │
   SequencePool (mean over sequence dim)  ◄────────────┘
     │
     ▼
   Linear(256 → num_classes)
     │
     ▼
   CrossEntropyLoss

**What it demonstrates:**

* Full transformer encoder architecture
* Embedding + PositionalEncoding for token representation
* Two stacked TransformerEncoderLayers
* SequencePool to collapse sequence dimension for classification
* End-to-end text classification pipeline with EnronSpam dataset

**Test command:**

.. code-block:: bash

   NNM_DIAGRAM=transformer_classifier pnpm run test:example

**Expected output:** Training on EnronSpam text classification with ~1 epoch.

---

Example Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 18 25 25

   * - Diagram
     - Architecture
     - Stereotypes Used
     - Key Concept
   * - mninst
     - MLP
     - Linear, ReLU, Flatten, CrossEntropyLoss
     - Sequential chain
   * - mnist_skips
     - MLP + skips
     - Addition
     - Skip connections
   * - autoencoder_mnist
     - Conv Autoencoder
     - Conv2d, MaxPool, Upsample, MSELoss
     - Encoder-decoder, reconstruction
   * - auto_encoder_submodels
     - Subflow AE
     - SubFlow
     - Container encapsulation
   * - auto_encoder_submodels_with_submodels
     - Nested Subflow
     - SubFlow (nested)
     - Recursive nesting
   * - single_head_attention
     - Attention
     - ScaledDotProduct, MatMul
     - Attention from primitives
   * - multihead_attention
     - Multi-head
     - Concat
     - Explicit multi-head
   * - horizontal_multihead_attention
     - Multi-head
     - HorizontalRepeat
     - vmap-based parallel
   * - skip_connections_with_repetition
     - Residual + Repeat
     - Fork, Addition, Repeat
     - Repeat + residual
   * - transformer_classifier
     - Transformer
     - Embedding, PositionalEncoding, TransformerEncoderLayer, SequencePool, Repeat
     - Full seq-to-seq pipeline
