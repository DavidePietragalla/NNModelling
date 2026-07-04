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

import { Diagram } from "../../Diagram.svelte";
import { NNTree, type NNTreeNode } from "../../conversion/nnTree";

/** Non-Input, non-Loss module stereotype names (for random graph generation). */
export const MODULE_STEREOS = [
  "Linear", "Conv2d", "ReLU", "Tanh", "Sigmoid", "Softmax", "Dropout",
  "BatchNorm1d", "BatchNorm2d", "LayerNorm", "Flatten", "MaxPool2d", "AvgPool2d",
  "Embedding", "Fork", "PositionalEncoding", "SequencePool",
];

export const JOIN_STEREOS = ["Addition", "Concat", "MatMul", "ScaledDotProduct", "Einsum", "MaskedScaledDotProduct"];

export const LOSS_STEREOS = ["CrossEntropyLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss"];

/** Create a clean Diagram and stub window. */
export function createDiagram(): Diagram {
  const d = new Diagram();
  return d;
}

/** Assert graph consistency invariants on a DiagramCore. */
export function assertGraphConsistent(d: Diagram): void {
  const ids = new Set(d.nodes.map(n => n.id));
  expect(ids.size).toBe(d.nodes.length);

  for (const e of d.edges) {
    expect(ids.has(e.source)).toBe(true);
    expect(ids.has(e.target)).toBe(true);
    expect(e.source).not.toBe(e.target);
  }

  for (const n of d.nodes) {
    if (n.parentId) expect(ids.has(n.parentId)).toBe(true);
  }

  expect(() => d.getSnapshot()).not.toThrow();
}

/** Assert NNTree invariants. */
export function assertTreeInvariants(tree: NNTree, d: Diagram): void {
  expect(tree.lossNode).not.toBeNull();
  const allIds = new Set(tree.nodes.keys());

  for (const [id, node] of tree.nodes) {
    for (const childId of node.children) {
      expect(allIds.has(childId) || (tree.lossNode && childId === tree.lossNode.name)).toBe(true);
    }
    if (node.data.type === "subflow") {
      const sf = node.data as import("../../conversion/nnTree").SubflowData;
      expect(typeof sf.entryNode).toBe("string");
      expect(sf.entryNode.length > 0).toBe(true);
      const innerIds = new Set(Object.keys(sf.nodes));
      expect(innerIds.has(sf.entryNode)).toBe(true);
    }
  }
}
