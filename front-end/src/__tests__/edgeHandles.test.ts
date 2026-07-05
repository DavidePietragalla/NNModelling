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

import { describe, it, expect, afterAll } from "vitest";
import { Diagram } from "../Diagram.svelte";
import type { StereotypeCore } from "../core/StereotypeCore";
import { stubWindow, unstubWindow } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

// ── Helpers ────────────────────────────────────────────────────────────

function getStereotype(diagram: Diagram, name: string): StereotypeCore {
  const s = diagram.getStereotype(name);
  if (!s) throw new Error(`Stereotype "${name}" not found`);
  return s;
}

// ── Edge Handle Contract Tests ─────────────────────────────────────────
// CONTRACT: CustomNode.svelte and DiagramCore.addEdge must agree on handle IDs.
//   CustomNode target Handle: id="in"
//   CustomNode source Handle: id="out"
//   DiagramCore.addEdge defaults: sourceHandle="out", targetHandle="in"
//
// BUG: CustomNode.svelte has NO explicit id attributes on its Handles.
//      SvelteFlow defaults to null when no id is set.
//      This means addEdge's "out"/"in" defaults don't match actual handles,
//      causing edges created via RPC to exist in state but not render.
//
// These tests verify the expected defaults from addEdge (documenting the contract).
// The actual fix must add id="in"/"out" to CustomNode.svelte to match.

describe("Edge handle ID defaults", () => {
  it("addEdge without handles defaults sourceHandle to 'out'", () => {
    const diagram = new Diagram();
    const linear = getStereotype(diagram, "Linear");

    diagram.addModule(linear, 0, 0);
    diagram.addModule(linear, 200, 0);
    const [src, tgt] = [diagram.nodes[1].id, diagram.nodes[2].id];

    const edge = diagram.addEdge(src, tgt);

    // These defaults must match CustomNode.svelte source Handle id
    expect(edge.sourceHandle).toBe("out");
    expect(edge.targetHandle).toBe("in");
  });

  it("addEdge with explicit targetHandle matches JoinNode handle convention", () => {
    const diagram = new Diagram();
    const linear = getStereotype(diagram, "Linear");
    const addition = getStereotype(diagram, "Addition");

    diagram.addModule(linear, 0, 0);
    diagram.addModule(linear, 0, 100);
    diagram.addJoinNode(addition, 200, 50, { inputsCount: 2 });
    const joinId = diagram.nodes[diagram.nodes.length - 1].id;

    // JoinNode handles are "in-0", "in-1", ..., "out"
    const edge0 = diagram.addEdge(diagram.nodes[1].id, joinId, undefined, "in-0");
    const edge1 = diagram.addEdge(diagram.nodes[2].id, joinId, undefined, "in-1");

    expect(edge0.targetHandle).toBe("in-0");
    expect(edge1.targetHandle).toBe("in-1");
  });

  it("RPC create_node + connect_nodes flow produces edge visible in state", () => {
    // Simulate the full MCP workflow:
    // 1. create_node (Linear) -> returns nodeId
    // 2. create_node (Linear) -> returns nodeId
    // 3. connect_nodes(source, target) -> returns edgeId
    // 4. verify edge exists in diagram.edges with correct handle IDs

    const diagram = new Diagram();
    const linear = getStereotype(diagram, "Linear");

    diagram.addModule(linear, 100, 100, { name: "NodeA" });
    diagram.addModule(linear, 300, 100, { name: "NodeB" });

    const nodeA = diagram.nodes[diagram.nodes.length - 2];
    const nodeB = diagram.nodes[diagram.nodes.length - 1];

    const edge = diagram.addEdge(nodeA.id, nodeB.id);

    // Edge exists in state
    expect(diagram.edges.length).toBeGreaterThanOrEqual(1);
    const found = diagram.edges.find((e) => e.id === edge.id);
    expect(found).toBeDefined();
    expect(found!.source).toBe(nodeA.id);
    expect(found!.target).toBe(nodeB.id);
    expect(found!.sourceHandle).toBe("out");
    expect(found!.targetHandle).toBe("in");
  });
});
