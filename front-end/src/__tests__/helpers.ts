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

import { type Node, type Edge } from "@xyflow/svelte";

/** Stub window so Diagram constructor can auto-spawn Input node. */
export function stubWindow() {
  (globalThis as any).window = { innerWidth: 1024 };
}

export function unstubWindow() {
  delete (globalThis as any).window;
}

/** Factory for test Node — 1 line instead of 5. */
export function node(
  id: string,
  stereotype: string,
  name: string,
  params: Record<string, { value: string; position?: string }> = {},
  overrides?: {
    color?: string;
    type?: string;
    isInput?: boolean;
    isLoss?: boolean;
    parentId?: string;
    hidden?: boolean;
  },
): Node {
  return {
    id,
    type: overrides?.type ?? "custom",
    position: { x: 0, y: 0 },
    parentId: overrides?.parentId,
    hidden: overrides?.hidden,
    data: {
      stereotype,
      name,
      color: overrides?.color ?? "#ccc",
      params: structuredClone(params),
      isInput: overrides?.isInput ?? false,
      isLoss: overrides?.isLoss ?? false,
    },
  } as Node;
}

/** Factory for test Edge. */
export function edge(
  id: string,
  source: string,
  target: string,
  handles?: { sourceHandle?: string; targetHandle?: string },
): Edge {
  return { id, source, target, ...handles } as Edge;
}
