Stereotypes Reference
=====================

Stereotypes define the behavior, appearance, and configurable parameters of
every node type in the visual editor. They are stored as JSON files in the
``Stereotypes/`` directory and loaded by ``StereotypeCore`` at runtime.

Categories
----------

Each stereotype has a ``category`` field that determines its role in the graph:

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Category
     - Role
   * - ``Input``
     - Entry point for data into the network
   * - ``Layer``
     - Standard neural network layer
   * - ``Activation``
     - Activation function
   * - ``Regularization``
     - Regularization (Dropout, etc.)
   * - ``Normalization``
     - Normalization layer (BatchNorm, LayerNorm)
   * - ``Shape``
     - Shape transformation (Flatten)
   * - ``Pooling``
     - Pooling operation (MaxPool, AvgPool, SequencePool)
   * - ``Embedding``
     - Embedding layer
   * - ``Attention``
     - Attention mechanism
   * - ``Transformer``
     - Transformer block
   * - ``Upsampling``
     - Upsampling operation (Upsample)
   * - ``Fork``
     - Passthrough node for explicit forks
   * - ``Position``
     - Positional encoding
   * - ``Loss``
     - Loss function (output node, no output handle)
   * - ``Join``
     - Multi-input merge node
   * - ``Subflow``
     - Container for sub-graphs with behavioral stereotypes

Modules (27)
------------

These are the standard neural network building blocks. Each module accepts one
input connection and has one output connection.

Linear
~~~~~~

.. code-block:: json

   {
     "category": "Layer",
     "pythonClassName": "nn.Linear",
     "color": "#4779c4"
   }

The standard fully-connected layer: ``y = xW^T + b``.

+------------------+--------+-----------+----------+
| Parameter        | Type   | Default   | Position |
+==================+========+===========+==========+
| ``in_features``  | int    | Undefined | top      |
+------------------+--------+-----------+----------+
| ``out_features`` | int    | Undefined | bottom   |
+------------------+--------+-----------+----------+
| ``bias``         | bool   | True      |          |
+------------------+--------+-----------+----------+

**Example:** Set ``in_features=784`` and ``out_features=128`` for the first
layer of an MNIST classifier.

Conv2d
~~~~~~

.. code-block:: json

   {
     "category": "Conv2d",
     "pythonClassName": "nn.Conv2d",
     "color": "#e67e22"
   }

2D convolution layer for image processing.

+--------------------+--------+-----------+----------+
| Parameter          | Type   | Default   | Position |
+====================+========+===========+==========+
| ``in_channels``    | int    | Undefined | top      |
+--------------------+--------+-----------+----------+
| ``out_channels``   | int    | Undefined | bottom   |
+--------------------+--------+-----------+----------+
| ``kernel_size``    | int    | Undefined |          |
+--------------------+--------+-----------+----------+
| ``stride``         | int    | 1         |          |
+--------------------+--------+-----------+----------+
| ``padding``        | int    | 0         |          |
+--------------------+--------+-----------+----------+
| ``bias``           | bool   | True      |          |
+--------------------+--------+-----------+----------+

**Example:** ``in_channels=1, out_channels=32, kernel_size=3`` for the first
convolutional layer of an MNIST autoencoder.

ReLU
~~~~

.. code-block:: json

   { "category": "Activation", "pythonClassName": "nn.ReLU", "color": "#2ecc71" }

Rectified Linear Unit: ``ReLU(x) = max(0, x)``. No parameters.

Tanh
~~~~

.. code-block:: json

   { "category": "Activation", "pythonClassName": "nn.Tanh", "color": "#2ecc71" }

Hyperbolic tangent activation: ``tanh(x)``. No parameters.

Sigmoid
~~~~~~~

.. code-block:: json

   { "category": "Activation", "pythonClassName": "nn.Sigmoid", "color": "#2ecc71" }

Sigmoid activation: ``σ(x) = 1 / (1 + e^{-x})``. No parameters.

Softmax
~~~~~~~

.. code-block:: json

   { "category": "Activation", "pythonClassName": "nn.Softmax", "color": "#2ecc71" }

Softmax activation for multi-class classification.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``dim``      | int    | Undefined |          |
+--------------+--------+-----------+----------+

Dropout
~~~~~~~

.. code-block:: json

   {
     "category": "Regularization",
     "pythonClassName": "nn.Dropout",
     "color": "#e74c3c"
   }

Randomly zeroes elements during training for regularization.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``p``        | float  | 0.5       |          |
+--------------+--------+-----------+----------+

**Example:** Set ``p=0.2`` for mild regularization.

BatchNorm1d
~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Normalization",
     "pythonClassName": "nn.BatchNorm1d",
     "color": "#1abc9c"
   }

Batch normalization for 1D (dense) features.

+-----------------+-------+----------+----------+
| Parameter       | Type  | Default  | Position |
+=================+=======+==========+==========+
| ``num_features``| int   | Undefined |         |
+-----------------+-------+----------+----------+

BatchNorm2d
~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Normalization",
     "pythonClassName": "nn.BatchNorm2d",
     "color": "#1abc9c"
   }

Batch normalization for 2D (convolutional) features.

+-----------------+-------+----------+----------+
| Parameter       | Type  | Default  | Position |
+=================+=======+==========+==========+
| ``num_features``| int   | Undefined |         |
+-----------------+-------+----------+----------+

LayerNorm
~~~~~~~~~

.. code-block:: json

   {
     "category": "Normalization",
     "pythonClassName": "nn.LayerNorm",
     "color": "#1abc9c"
   }

Layer normalization.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``normalized_shape`` | int | Undefined |     |
+--------------+--------+-----------+----------+

Flatten
~~~~~~~

.. code-block:: json

   {
     "category": "Shape",
     "pythonClassName": "nn.Flatten",
     "color": "#95a5a6"
   }

Flattens all dimensions starting from ``start_dim`` (default 1).

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``start_dim``| int    | 1         |          |
+--------------+--------+-----------+----------+

**Note:** Flatten is explicit in NNModelling — there is no auto-flatten
heuristic in the forward pass. You must insert a Flatten node when
transitioning from convolutional to linear layers.

MaxPool2d
~~~~~~~~~

.. code-block:: json

   {
     "category": "Pooling",
     "pythonClassName": "nn.MaxPool2d",
     "color": "#95a5a6"
   }

2D max pooling.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``kernel_size`` | int | Undefined |          |
+--------------+--------+-----------+----------+
| ``stride``   | int    | Undefined |          |
+--------------+--------+-----------+----------+

AvgPool2d
~~~~~~~~~

.. code-block:: json

   {
     "category": "Pooling",
     "pythonClassName": "nn.AvgPool2d",
     "color": "#95a5a6"
   }

2D average pooling. Same parameters as MaxPool2d.

Embedding
~~~~~~~~~

.. code-block:: json

   {
     "category": "Embedding",
     "pythonClassName": "nn.Embedding",
     "color": "#3498db"
   }

Token embedding lookup table.

+--------------------+--------+-----------+----------+
| Parameter          | Type   | Default   | Position |
+====================+========+===========+==========+
| ``num_embeddings`` | int    | Undefined | top      |
+--------------------+--------+-----------+----------+
| ``embedding_dim``  | int    | Undefined | bottom   |
+--------------------+--------+-----------+----------+

**Example:** ``num_embeddings=10000, embedding_dim=256`` for a vocabulary
of 10,000 tokens with 256-dimensional embeddings.

MultiheadAttention
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Attention",
     "pythonClassName": "nn.MultiheadAttention",
     "color": "#9b59b6"
   }

Multi-head attention as defined in "Attention is All You Need".

+--------------------+--------+-----------+----------+
| Parameter          | Type   | Default   | Position |
+====================+========+===========+==========+
| ``embed_dim``      | int    | Undefined |          |
+--------------------+--------+-----------+----------+
| ``num_heads``      | int    | Undefined |          |
+--------------------+--------+-----------+----------+
| ``dropout``        | float  | 0.0       |          |
+--------------------+--------+-----------+----------+
| ``batch_first``    | bool   | True      |          |
+--------------------+--------+-----------+----------+

**Note:** ``batch_first=True`` by default (PyTorch convention).

Transformer
~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Transformer",
     "pythonClassName": "nn.Transformer",
     "color": "#9b59b6"
   }

Full Transformer model (encoder + decoder with cross-attention).

+-----------------------+-------+----------+----------+
| Parameter             | Type  | Default  | Position |
+=======================+=======+==========+==========+
| ``d_model``           | int   | 512      |          |
+-----------------------+-------+----------+----------+
| ``nhead``             | int   | 8        |          |
+-----------------------+-------+----------+----------+
| ``num_encoder_layers``| int   | 6        |          |
+-----------------------+-------+----------+----------+
| ``num_decoder_layers``| int   | 6        |          |
+-----------------------+-------+----------+----------+
| ``dim_feedforward``   | int   | 2048     |          |
+-----------------------+-------+----------+----------+
| ``dropout``           | float | 0.1      |          |
+-----------------------+-------+----------+----------+

TransformerEncoderLayer
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Transformer",
     "pythonClassName": "nn.TransformerEncoderLayer",
     "color": "#9b59b6"
   }

Single encoder layer (self-attention + feedforward). Same parameters as
``Transformer`` without decoder-specific ones.

TransformerDecoderLayer
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Transformer",
     "pythonClassName": "nn.TransformerDecoderLayer",
     "color": "#9b59b6"
   }

Single decoder layer (self-attention + cross-attention + feedforward).

Upsample
~~~~~~~~

.. code-block:: json

   {
     "category": "Upsampling",
     "pythonClassName": "nn.Upsample",
     "color": "#e67e22"
   }

Upsamples a tensor to a given size.

+--------------------+--------+-----------+----------+
| Parameter          | Type   | Default   | Position |
+====================+========+===========+==========+
| ``scale_factor``   | float  | Undefined |          |
+--------------------+--------+-----------+----------+
| ``mode``           | str    | nearest   |          |
+--------------------+--------+-----------+----------+

**Example:** ``scale_factor=2, mode='bilinear'`` for 2x upsampling in a
decoder/autoencoder.

Fork
~~~~

.. code-block:: json

   {
     "category": "Fork",
     "pythonClassName": "",
     "color": "#7f8c8d"
   }

A passthrough node that forwards its input unchanged. Used to make forks
explicit inside subflows (e.g. residual connections in a Repeat block).
No parameters.

**Example:** Inside a residual block, use Fork to split the path — one
branch goes through the weight layers, the other skips directly to the
Addition join.

PositionalEncoding
~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Position",
     "pythonClassName": "ops.PositionalEncoding",
     "color": "#3498db"
   }

Sinusoidal positional encoding table (not learned).

+--------------------+--------+-----------+----------+
| Parameter          | Type   | Default   | Position |
+====================+========+===========+==========+
| ``d_model``        | int    | Undefined |          |
+--------------------+--------+-----------+----------+
| ``max_len``        | int    | 5000      |          |
+--------------------+--------+-----------+----------+

**Example:** Used after Embedding in a transformer classifier to add
position information to token embeddings.

SequencePool
~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Pooling",
     "pythonClassName": "ops.SequencePool",
     "color": "#95a5a6"
   }

Mean pooling over the sequence dimension. No parameters.

**Example:** Used after transformer encoder layers to collapse the sequence
into a single vector for classification.

Loss Functions (4)
~~~~~~~~~~~~~~~~~~

Loss nodes are **output nodes** — they have no output handle and define
the task type for metric selection.

.. list-table::
   :header-rows: 1
   :widths: 25 30 15

   * - Stereotype
     - pythonClassName
     - Category
   * - ``BCELoss``
     - ``nn.BCELoss``
     - Loss
   * - ``BCEWithLogitsLoss``
     - ``nn.BCEWithLogitsLoss``
     - Loss
   * - ``CrossEntropyLoss``
     - ``nn.CrossEntropyLoss``
     - Loss
   * - ``MSELoss``
     - ``nn.MSELoss``
     - Loss

The loss node connected to the graph determines:

* **Task type** — ``CrossEntropyLoss`` and ``BCEWithLogitsLoss`` set
  classification, ``MSELoss`` sets regression
* **Metrics** — Classification uses ``Accuracy``, regression uses ``MSE``
* **Loss computation** — The loss function used during training

Joins (6)
---------

Join nodes accept multiple inputs and produce a single output. They are the
explicit mechanism for merging branches in the data flow.

Addition
~~~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.Addition" }

Element-wise sum of all inputs. All input tensors must have the same shape.
No parameters.

**Usage:** Skip connections (residual networks), combining feature paths.

Concat
~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.Concat" }

Concatenates tensors along a specified dimension.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``dim``      | int    | -1        |          |
+--------------+--------+-----------+----------+

**Example:** ``dim=1`` concatenates along the channel dimension in a
``[batch, channels, h, w]`` tensor — used to combine multi-head attention
outputs.

Einsum
~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.Einsum" }

Einstein summation over arbitrary tensors.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``equation`` | str    | Undefined |          |
+--------------+--------+-----------+----------+

**Example:** ``equation='bij,bjk->bik'`` for batched matrix multiplication.

MatMul
~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.MatMul" }

Matrix multiplication: ``inputs[0] @ inputs[1]``. Requires exactly 2 inputs.
No parameters.

**Usage:** Simple attention implementations, projection operations.

ScaledDotProduct
~~~~~~~~~~~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.ScaledDotProduct" }

Scaled dot-product attention: ``softmax(Q @ K^T / sqrt(d_k)) @ V``.

Expects 3 inputs in order: Query, Key, Value. No parameters.

**Usage:** Building attention mechanisms from primitives (see
``single_head_attention.json`` example).

MaskedScaledDotProduct
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   { "category": "Join", "pythonClassName": "ops.MaskedScaledDotProduct" }

Same as ScaledDotProduct but with an optional causal mask (upper triangular).
Expects 3 inputs (Q, K, V) plus an optional mask input.

**Usage:** Causal self-attention in decoders and autoregressive models.

SubFlows (2)
------------

Subflow stereotypes define **containers** that hold sub-graphs and apply
structural modifications.

Repeat
~~~~~~

.. code-block:: json

   {
     "category": "Subflow",
     "pythonClassName": "ops.Repeat",
     "color": "#9b59b6"
   }

Creates N sequential copies of the internal sub-graph, wrapped in
``nn.Sequential``. Each copy has **independent weights**.

+----------------+--------+-----------+----------+
| Parameter      | Type   | Default   | Position |
+================+========+===========+==========+
| ``iterations`` | int    | 1         | top      |
+----------------+--------+-----------+----------+

**Example:** Set ``iterations=3`` inside a residual block to create 3
sequential residual layers with independent weights.

**Use case:** Building deep residual networks, repeated transformer blocks.

HorizontalRepeat
~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Subflow",
     "pythonClassName": "ops.HorizontalRepeat",
     "color": "#9b59b6"
   }

Creates N parallel copies of the internal sub-graph using **vmap** and
**functional_call** with **stacked module state**. The outputs are
concatenated along the last dimension.

+--------------+--------+-----------+----------+
| Parameter    | Type   | Default   | Position |
+==============+========+===========+==========+
| ``n``        | int    | 4         | top      |
+--------------+--------+-----------+----------+

**Example:** Set ``n=4`` around a single-head attention subflow to create a
4-head attention mechanism (see ``horizontal_multihead_attention.json``).

**Important:** The join after HorizontalRepeat is hardcoded to concat on
``dim=-1``. This is not configurable — the output shape becomes
``[batch, ..., n * d]`` where ``d`` is the output dimension of a single head.

Stereotype JSON Format
----------------------

Each stereotype JSON follows this structure:

.. code-block:: json

   {
     "category": "Layer",              // Determines node behavior
     "pythonClassName": "nn.Linear",   // Maps to Python class or ops module
     "expr": "",                       // Expression language (future)
     "view": {
       "color": "#4779c4",             // Default node color
       "width": 140,                   // Default node width
       "height": 60                    // Default node height
     },
     "params": {
       "parameter_name": {
         "type": "int",                // Parameter type (int, float, bool, str)
         "default": "Undefined",       // Default value
         "position": "top"             // Display position: top, bottom, or omit
       }
     }
   }

The ``position`` field controls where the parameter is displayed on the node:

* ``"top"`` — shown in the top area of the node (e.g., ``in_features``)
* ``"bottom"`` — shown in the bottom area (e.g., ``out_features``)
* (omitted) — rendered inline in the sidebar
