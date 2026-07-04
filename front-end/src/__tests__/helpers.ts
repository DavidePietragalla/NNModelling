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
