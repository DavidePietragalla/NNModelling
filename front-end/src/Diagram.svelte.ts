/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
