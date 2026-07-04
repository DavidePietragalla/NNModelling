/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

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
    // Clear the undo snapshot captured during auto-spawn of Input node —
    // the initial Input should not be undoable.
    this._undoStack = [];
    this._redoStack = [];
  }
}
