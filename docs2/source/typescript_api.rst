TypeScript / Frontend API Reference
====================================

The frontend TypeScript API is documented using **TypeDoc**. The generated
documentation is available as a static site.

Viewing the TypeScript API
--------------------------

After building the documentation (see :doc:`user_guide`), open:

.. code-block:: text

   docs2/build/typedoc/index.html

This page provides full API documentation for all exported symbols, including:

* ``DiagramCore`` — main state authority with all graph manipulation methods
* ``StereotypeCore`` — browser stereotype JSON loader (Vite ``import.meta.glob``)
* ``BrowserRPCHandler`` — WebSocket RPC handler for MCP integration
* ``checkValidConnection`` — standalone connection validation
* ``findDirectedCycle`` — cycle detection over directed graph edges
* ``Position``, ``NodeConfig``, ``JoinNodeConfig``, ``DiagramCoreSnapshot`` —
  core type definitions

Generating the TypeScript API
-----------------------------

To regenerate the TypeDoc output:

.. code-block:: bash

   cd front-end
   pnpm exec typedoc --options typedoc.json

Or as part of the full documentation build:

.. code-block:: bash

   # From project root
   pnpm run docs:typedoc

Architecture Overview
---------------------

The TypeScript codebase is organized into two layers:

**core/** (Pure TypeScript — no Svelte dependencies)
    These modules can run in any JavaScript environment and are the foundation
    of the editor's business logic:

    * ``DiagramCore`` — manages nodes, edges, undo/redo, import/export, and
      exposes the synchronous ``onGraphChanged`` graph-change subscription
    * ``StereotypeCore`` — loads stereotype JSON from the ``Stereotypes/``
      directory via Vite's ``import.meta.glob`` (browser)
    * ``types.ts`` — shared type definitions (``Position``, ``NodeConfig``,
      ``JoinNodeConfig``, ``DiagramCoreSnapshot``)
    * ``validation.ts`` — connection validation rules and cycle detection
      (``checkValidConnection``, ``findDirectedCycle``)

**sync/** (Browser-Side RPC)
    * ``BrowserRPCHandler`` — handles WebSocket JSON-RPC requests from the
      MCP server, executing methods on the local ``DiagramCore``

Key Types
---------

The ``core`` barrel exports these configuration and snapshot types (the
``types.ts`` module also re-exports ``Node`` and ``Edge`` from Svelte Flow):

.. code-block:: typescript

   interface Position {
     x: number;
     y: number;
   }

   interface NodeConfig {
     name?: string;
     color?: string;
     width?: number;
     height?: number;
     params?: Record<string, any>;
   }

   interface JoinNodeConfig extends NodeConfig {
     inputsCount?: number;
   }

   interface DiagramCoreSnapshot {
     nodes: Node[];
     edges: Edge[];
   }

``DiagramCore`` also exposes the synchronous ``onGraphChanged(handler)``
graph-change subscription: the handler runs once after every successful public
mutation (add/update/delete/move operations, edge changes, undo/redo, snapshot
restore, import and reset), carries no payload, and returns an unsubscribe
function. Rejected connections and no-op operations do not notify.

For complete type signatures and method documentation, refer to the
TypeDoc output linked above.
