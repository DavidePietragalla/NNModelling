Python API Reference
====================

This reference is auto-generated from Python docstrings using Sphinx's
``autodoc`` extension. All modules live under ``converted/src/``.

.. code-block:: text

   converted/src/
   ├── __init__.py
   ├── convert.py          NNTree JSON → Hydra YAML configs
   ├── main.py             Training entry point
   ├── infer.py            Inference entry point
   ├── net/
   │   └── base.py         Dynamic DAG LightningModule
   ├── ops/                Custom operations (11 modules)
   └── dataset/            Dataset classes (3 modules)

Network Core
------------

.. automodule:: net.base
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Joins and subflow operations that are instantiated during the forward pass.

### Addition (element-wise sum)

.. automodule:: ops.addition
   :members:
   :undoc-members:

### Concat (torch.cat)

.. automodule:: ops.concat
   :members:
   :undoc-members:

### Einsum (Einstein summation)

.. automodule:: ops.einsum
   :members:
   :undoc-members:

### MatMul (matrix multiplication)

.. automodule:: ops.mat_mul
   :members:
   :undoc-members:

### ScaledDotProduct (attention)

.. automodule:: ops.scaled_dot_product
   :members:
   :undoc-members:

### MaskedScaledDotProduct (causal attention)

.. automodule:: ops.masked_scaled_dot_product
   :members:
   :undoc-members:

### Subflow (BFS internal graph executor)

.. automodule:: ops.subflow
   :members:
   :undoc-members:

### Repeat (N sequential copies)

.. automodule:: ops.repeat
   :members:
   :undoc-members:

### HorizontalRepeat (N parallel copies via vmap)

.. automodule:: ops.horizontal_repeat
   :members:
   :undoc-members:

### PositionalEncoding

.. automodule:: ops.positional_encoding
   :members:
   :undoc-members:

### SequencePool

.. automodule:: ops.sequence_pool
   :members:
   :undoc-members:

Datasets
--------

### MNIST (image classification)

.. automodule:: dataset.mnist
   :members:
   :undoc-members:

### Autoencoder MNIST

.. automodule:: dataset.autoencoder_mnist
   :members:
   :undoc-members:

### EnronSpam (text classification)

.. automodule:: dataset.enron_spam
   :members:
   :undoc-members:

Conversion Pipeline
-------------------

### convert.py — NNTree to Hydra configs

.. automodule:: convert
   :members:
   :undoc-members:

### main.py — Training entry point

.. automodule:: main
   :members:
   :undoc-members:

### infer.py — Inference entry point

.. automodule:: infer
   :members:
   :undoc-members:
