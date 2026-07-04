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
* ``EventBus`` — typed event emitter with monotonic sequencing
* ``StereotypeCore`` — dual loader for stereotype JSON files
* ``BrowserRPCHandler`` — WebSocket RPC handler for MCP integration
* ``checkValidConnection`` — standalone connection validation
* ``DomainEvent`` and related types — event system type definitions

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

    * ``DiagramCore`` — manages nodes, edges, undo/redo, import/export
    * ``EventBus`` — typed event system for mutation tracking
    * ``StereotypeCore`` — loads stereotype JSON from Vite (browser) or
      Node.js ``fs`` (MCP server)
    * ``types.ts`` — shared type definitions
    * ``validation.ts`` — connection validation rules

**sync/** (Browser-Side RPC)
    * ``BrowserRPCHandler`` — handles WebSocket JSON-RPC requests from the
      MCP server, executing methods on the local ``DiagramCore``

Key Types
---------

.. code-block:: typescript

   // Domain events for tracking mutations
   interface DomainEvent<T extends string> {
     seq: number;
     type: T;
     timestamp: number;
     payload: unknown;
   }

   // WebSocket protocol messages
   interface WSSnapshotMessage {
     type: "snapshot";
     state: { nodes: Node[]; edges: Edge[] };
   }

   interface WSDeltaMessage {
     type: "delta";
     operations: DeltaOperation[];
   }

   // Config interfaces for Hydra pipeline
   interface AppConfig {
     net: NetConfig;
     dataset: DatasetConfig;
     optimizer: OptimizerConfig;
     trainer: TrainerConfig;
   }

For complete type signatures and method documentation, refer to the
TypeDoc output linked above.
