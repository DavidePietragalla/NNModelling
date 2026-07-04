Stereotypes Reference
=====================

Stereotypes define the behavior, appearance, and configurable parameters of
every node in the visual editor. They are stored as JSON files in
``Stereotypes/`` and loaded by ``StereotypeCore`` at runtime.

Each stereotype is a plain JSON file that maps a visual node to a Python class,
defines its input/output behavior, and declares its configurable parameters.

Categories
----------

The ``category`` field determines the node's role in the graph and its handle
configuration:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Role
     - Handles
   * - ``Input``
     - Network entry point
     - 0 input, 1 output
   * - ``Fork``
     - Passthrough for explicit branching inside subflows
     - 1 input, 1 output
   * - ``Layer``
     - Standard module (Linear, Conv2d, ReLU, Dropout, ...)
     - 1 input, 1 output
   * - ``Loss``
     - Loss function / output node (BCELoss, CrossEntropyLoss, ...)
     - 1 input, 0 output
   * - ``Join``
     - Multi-input merge node (Addition, Concat, MatMul, ...)
     - N inputs, 1 output
   * - ``Subflow``
     - Container holding a sub-graph with structural transformation
     - 1 input, 1 output
   * - ``Module``
     - Generic; reserved for future use, currently unused
     - Depends on implementation

JSON Field Reference
--------------------

Every stereotype JSON can use these fields:

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 10 55

   * - Field
     - Type
     - Required
     - Default
     - Description
   * - ``category``
     - string
     - yes
     -
     - One of: ``Input``, ``Fork``, ``Layer``, ``Loss``, ``Join``, ``Subflow``, ``Module``
   * - ``pythonClassName``
     - string
     - yes
     -
     - Fully qualified Python class path, e.g. ``nn.Linear``, ``ops.Addition``, ``ops.Repeat``. Set to ``""`` or ``"None"`` for nodes with no Python counterpart (Input, Fork).
   * - ``expr``
     - string
     - no
     - ``""``
     - Expression language string for defining custom behavior at the node level. Reserved for future use.
   * - ``taskType``
     - string
     - no
     -
     - Forces a task type (``classification``, ``regression``). Only meaningful on ``Loss`` nodes — overrides automatic detection.
   * - ``view.color``
     - string
     - no
     - ``#cccccc``
     - Default hex color for the node in the editor.
   * - ``view.width``
     - number
     - no
     - ``140``
     - Default node width in pixels.
   * - ``view.height``
     - number
     - no
     - ``60``
     - Default node height in pixels.
   * - ``params``
     - object
     - no
     - ``{}``
     - Map of parameter names to their definitions. Each key is the parameter name.
   * - ``params.<name>.type``
     - string
     - yes
     -
     - Type of the parameter: ``int``, ``float``, ``bool``, ``str``, ``Tensor``.
   * - ``params.<name>.default``
     - string
     - no
     - ``"Undefined"``
     - Default value as a string (parsed by ``ast.literal_eval`` on the Python side).
   * - ``params.<name>.position``
     - string
     - no
     -
     - Display position on the node: ``"top"``, ``"bottom"``, or omit for sidebar-only.

Examples
--------

Three representative stereotypes showing the full range of features:

Layer: Linear
~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Layer",
     "pythonClassName": "nn.Linear",
     "expr": "",
     "view": {
       "color": "#4779c4",
       "width": 140,
       "height": 60
     },
     "params": {
       "in_features": {
         "type": "int",
         "default": "Undefined",
         "position": "top"
       },
       "out_features": {
         "type": "int",
         "default": "Undefined",
         "position": "bottom"
       },
       "bias": {
         "type": "bool",
         "default": "True"
       }
     }
   }

The most common stereotype pattern: ``category: "Layer"`` maps a visual node to
a ``nn.Module`` subclass. Parameters with ``position: "top"`` are displayed on
the node's upper half, ``position: "bottom"`` on the lower half. Parameters
without a position appear in the sidebar only.

Join: Concat
~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Join",
     "pythonClassName": "ops.Concat",
     "view": {
       "color": "#2ecc71",
       "width": 80,
       "height": 60
     },
     "params": {
       "dim": {
         "type": "int",
         "default": "-1",
         "position": "top"
       }
     }
   }

Join nodes accept multiple incoming connections (their ``inputsCount`` defaults
to 2 and can be incremented in the UI). The ``pythonClassName`` points to an
``ops.*`` module that implements the merge logic. Input order is preserved from
edge ``targetHandle`` labels (``"in-0"``, ``"in-1"``, ...).

Subflow: Repeat
~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "category": "Subflow",
     "pythonClassName": "ops.Repeat",
     "expr": "",
     "view": {
       "color": "#9b59b6",
       "width": 400,
       "height": 300
     },
     "params": {
       "iterations": {
         "type": "int",
         "default": "1",
         "position": "top"
       }
     }
   }

Subflow stereotypes define **containers** that hold an entire sub-graph (nodes
with ``parentId`` set to the subflow node). The ``pythonClassName`` references
a structural operation that runs or transforms the sub-graph. Subflows use
``_recursive_: false`` in the generated Hydra config to prevent recursive
instantiation.

Notes
-----

* **Loss nodes** determine task type for metric selection: ``CrossEntropyLoss``
  and ``BCEWithLogitsLoss`` set classification, ``MSELoss`` sets regression.
  They have no output handle — they are terminal nodes.

* **Flatten is explicit**: there is no auto-flatten heuristic in the forward
  pass. You must insert a Flatten node when transitioning from convolutional
  to linear layers.

* **HorizontalRepeat** has its join hardcoded to concat on ``dim=-1``. Output
  shape becomes ``[batch, ..., n * d]``. This is not configurable.

* **Join input order** matters for non-commutative joins like ``MatMul`` and
  ``ScaledDotProduct``. The order is determined by the ``targetHandle`` labels
  (``"in-0"``, ``"in-1"``, ...), not by BFS traversal.

* **Parameter display positions**: ``"top"`` renders above the node's center,
  ``"bottom"`` below, omitted renders in the sidebar's parameter panel only.
