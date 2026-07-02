// front-end/src/Diagram.svelte.ts
// Thin Svelte wrapper around DiagramCore.
// Adds $state.raw reactivity for Svelte 5.

import { type Node, type Edge } from "@xyflow/svelte";
import { DiagramCore } from "./core/DiagramCore";
import { Stereotype } from "./stereotype";

export class Diagram extends DiagramCore {
  public nodes: Node[] = $state.raw<Node[]>([]);
  public edges: Edge[] = $state.raw<Edge[]>([]);

  constructor() {
    super();
    this.initStereotypes(Stereotype.loadFromDirectory());
    const inputStereotype = this.stereotypes.find(s => s.isInput);
    if (inputStereotype && this.nodes.length === 0) {
      const centerX = (typeof window !== "undefined" ? window.innerWidth : 1024) / 2 - 15;
      this.addModule(inputStereotype, centerX, 50);
    }
  }
}
