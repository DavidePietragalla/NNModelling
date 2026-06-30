// front-end/src/Diagram.svelte.ts
// Thin Svelte wrapper around DiagramCore.
// Adds $state.raw reactivity for Svelte 5 and re-hydrates Svelte-specific
// callbacks (onToggle, onResizeEnd) that are lost during JSON serialization.

import { type Node, type Edge } from "@xyflow/svelte";
import { DiagramCore } from "./core/DiagramCore";
import { Stereotype } from "./stereotype";

export class Diagram extends DiagramCore {
  // Override DiagramCore's plain arrays with Svelte 5 $state.raw for reactivity.
  // When DiagramCore methods do this.nodes = [...], they write to the $state.raw
  // version here, triggering Svelte 5 reactive updates in FlowCanvas.svelte.
  public nodes: Node[] = $state.raw<Node[]>([]);
  public edges: Edge[] = $state.raw<Edge[]>([]);

  constructor() {
    super();

    // Load stereotypes using Vite's import.meta.glob (compile-time),
    // inject them into DiagramCore via initStereotypes()
    this.initStereotypes(Stereotype.loadFromDirectory());

    // Auto-spawn Input node (Svelte-specific behavior)
    const inputStereotype = this.stereotypes.find(s => s.isInput);
    if (inputStereotype && this.nodes.length === 0) {
      const centerX = (typeof window !== "undefined" ? window.innerWidth : 1024) / 2 - 15;
      this.addModule(inputStereotype, centerX, 50);
    }
  }

  // ── Serialization (overrides) ────────────────────────────────────
  //
  // importFromJson is overridden to re-hydrate Svelte-specific callbacks
  // (onToggle, onResizeEnd) that are lost during JSON.stringify/parse.
  // The base DiagramCore.importFromJson stores only plain data.

  public importFromJson(jsonString: string) {
    try {
      const parsedData = JSON.parse(jsonString);

      // Validate expected structure
      if (Array.isArray(parsedData.nodes) && Array.isArray(parsedData.edges)) {
        // Re-hydrate subflow callbacks (Svelte-specific)
        const hydratedNodes = parsedData.nodes.map((n: Node) => {
          if (n.type === "subflow") {
            return {
              ...n,
              data: {
                ...n.data,
                onToggle: (id: string, collapse: boolean) => this.toggleSubflow(id, collapse),
                onResizeEnd: (nodeId: string, w: number, h: number) => {
                  this.nodes = this.nodes.map((node) => {
                    if (node.id === nodeId && !node.data.isCollapsed) {
                      return { ...node, data: { ...node.data, oldWidth: w, oldHeight: h } } as Node;
                    }
                    return node;
                  });
                },
              },
            } as Node;
          }
          return n;
        });
        this.nodes = hydratedNodes;
        this.edges = parsedData.edges;

        // Emit import event
        this.events.emit("diagram_imported", { nodes: this.nodes, edges: this.edges });
        this.events.emit("graph_changed", {
          nodeCount: this.nodes.length,
          edgeCount: this.edges.length,
        });
      } else {
        throw new Error("Il file JSON non contiene un formato valido (nodi o edges mancanti).");
      }
    } catch (error) {
      console.error("Errore durante l'importazione del modello:", error);
      alert("Errore: Impossibile caricare il file. Verifica che sia un JSON valido.");
    }
  }

  // ── Subflow creation (overrides) ─────────────────────────────────
  //
  // addSubGraph is overridden to include Svelte-specific callbacks
  // (onToggle, onResizeEnd) in node data. The base DiagramCore.addSubGraph
  // stores only plain data without callbacks.

  public addSubGraph(x: number, y: number) {
    const id = `subflow_${Date.now()}`;
    const newSubgraph: Node = {
      id,
      type: "subflow",
      position: { x, y },
      data: {
        label: `${id}`,
        isCollapsed: false,
        onToggle: (subflowId: string, collapse: boolean) => this.toggleSubflow(subflowId, collapse),
        oldWidth: 400,
        oldHeight: 300,
        onResizeEnd: (nodeId: string, w: number, h: number) => {
          this.nodes = this.nodes.map(n => {
            if (n.id === nodeId && !n.data.isCollapsed) {
              return {
                ...n,
                data: { ...n.data, oldWidth: w, oldHeight: h }
              } as Node;
            }
            return n;
          });
        }
      },
      width: 400,
      height: 300
    };
    this.nodes = [...this.nodes, newSubgraph];

    this.events.emit("node_created", {
      nodeId: newSubgraph.id,
      name: newSubgraph.id,
      type: "subflow",
      stereotype: "Subflow"
    });
    this.events.emit("graph_changed", {
      nodeCount: this.nodes.length,
      edgeCount: this.edges.length,
    });
  }
}
