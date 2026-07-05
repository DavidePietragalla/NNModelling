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
import { stubWindow, unstubWindow } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Undo/Redo tests
// ---------------------------------------------------------------------------
// The Diagram constructor auto-spawns one Input node (isInput).
// After construction, undo stack is cleared (Issue 4 fix),
// so initial state = exactly 1 node (Input), empty undo/redo stacks.
// ---------------------------------------------------------------------------

function getLinearStereotype(diagram: Diagram) {
  const s = diagram.getStereotype("Linear");
  if (!s) throw new Error("Linear stereotype not found");
  return s;
}

describe("Diagram undo/redo", () => {
  it("1. Undo after addModule — node count drops back", () => {
    const diagram = new Diagram();
    expect(diagram.nodes.length).toBe(1); // Input node auto-spawn

    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    expect(diagram.nodes.length).toBe(2);

    const ok = diagram.undo();
    expect(ok).toBe(true);
    expect(diagram.nodes.length).toBe(1);
  });

  it("2. Redo after undo — node comes back", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    expect(diagram.nodes.length).toBe(2);

    diagram.undo();
    expect(diagram.nodes.length).toBe(1);

    const ok = diagram.redo();
    expect(ok).toBe(true);
    expect(diagram.nodes.length).toBe(2);
  });

  it("3. Undo after deleteNodes — nodes restored", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    diagram.addModule(getLinearStereotype(diagram), 200, 200);
    expect(diagram.nodes.length).toBe(3); // Input + 2 Linear

    // Delete the two non-Input nodes
    const idsToDelete = diagram.nodes
      .filter((n) => !n.data.isInput)
      .map((n) => n.id);
    diagram.deleteNodes(idsToDelete);
    expect(diagram.nodes.length).toBe(1);

    const ok = diagram.undo();
    expect(ok).toBe(true);
    expect(diagram.nodes.length).toBe(3);
  });

  it("4. Redo stack cleared by new action — redo() returns false", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    diagram.undo();
    expect(diagram.nodes.length).toBe(1);
    expect(diagram.redo()).toBe(true); // sanity: redo works before new action
    expect(diagram.nodes.length).toBe(2);

    // Now: add another node (undo again first to get back to 1)
    diagram.undo();
    expect(diagram.nodes.length).toBe(1);
    // New action
    diagram.addModule(getLinearStereotype(diagram), 300, 300);
    expect(diagram.nodes.length).toBe(2);

    // Redo stack should be cleared — redo() returns false
    expect(diagram.redo()).toBe(false);
    expect(diagram.nodes.length).toBe(2);
  });

  it("5. Multiple undos — back to initial state", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    diagram.addModule(getLinearStereotype(diagram), 200, 200);
    diagram.addModule(getLinearStereotype(diagram), 300, 300);
    expect(diagram.nodes.length).toBe(4); // Input + 3 Linear

    // Undo 3 times — each undoes one addModule
    expect(diagram.undo()).toBe(true);
    expect(diagram.undo()).toBe(true);
    expect(diagram.undo()).toBe(true);
    expect(diagram.nodes.length).toBe(1);

    // No more undos
    expect(diagram.undo()).toBe(false);
  });

  it("6. Undo limit at 50 entries", () => {
    const diagram = new Diagram();
    const stereo = getLinearStereotype(diagram);

    // Perform 55 separate addModule actions
    for (let i = 0; i < 55; i++) {
      diagram.addModule(stereo, i * 10, i * 10);
    }
    // 56 nodes total = 1 Input + 55 Linear
    expect(diagram.nodes.length).toBe(56);

    // Undo stack should have at most 50 entries (capped in _captureUndoState)
    const stackLen = (diagram as any)._undoStack.length;
    expect(stackLen).toBeLessThanOrEqual(50);

    // Undo 50 times — each succeeds
    for (let i = 0; i < 50; i++) {
      expect(diagram.undo()).toBe(true);
    }

    // 51st undo returns false
    expect(diagram.undo()).toBe(false);
  });

  it("7. Undo after addEdge — edge removed", () => {
    const diagram = new Diagram();
    // Add two modules to connect
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    diagram.addModule(getLinearStereotype(diagram), 200, 200);
    const nonInput = diagram.nodes.filter((n) => !n.data.isInput);
    expect(nonInput.length).toBe(2);

    // Connect nonInput[0] → nonInput[1]
    diagram.addEdge(nonInput[0].id, nonInput[1].id);
    expect(diagram.edges.length).toBe(1);

    // Undo should remove the edge
    const ok = diagram.undo();
    expect(ok).toBe(true);
    expect(diagram.edges.length).toBe(0);
  });

  it("8. Undo after updateModule — params restored", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    const node = diagram.nodes.find((n) => !n.data.isInput)!;

    // Original params (empty, since addModule passes no customConfig.params)
    const originalParams: unknown = node.data.params;

    // Update params
    diagram.updateModule(node.id, {
      params: { out_features: { value: "999" } },
    });
    const updatedNode = diagram.nodes.find((n) => n.id === node.id)!;
    expect(updatedNode.data.params).not.toEqual(originalParams);

    // Undo should restore original params
    const ok = diagram.undo();
    expect(ok).toBe(true);
    const restoredNode = diagram.nodes.find((n) => n.id === node.id)!;
    expect(restoredNode.data.params).toEqual(originalParams);
  });

  it("9. Undo after moveNode — position restored", () => {
    const diagram = new Diagram();
    diagram.addModule(getLinearStereotype(diagram), 100, 100);
    const node = diagram.nodes.find((n) => !n.data.isInput)!;

    expect(node.position.x).toBe(100);
    expect(node.position.y).toBe(100);

    // Move node
    diagram.moveNode(node.id, 500, 600);
    const movedNode = diagram.nodes.find((n) => n.id === node.id)!;
    expect(movedNode.position.x).toBe(500);
    expect(movedNode.position.y).toBe(600);

    // Undo should restore original position
    const ok = diagram.undo();
    expect(ok).toBe(true);
    const restoredNode = diagram.nodes.find((n) => n.id === node.id)!;
    expect(restoredNode.position.x).toBe(100);
    expect(restoredNode.position.y).toBe(100);
  });

  it("10. Undo after toggleSubflow — subflow expanded again", () => {
    const diagram = new Diagram();

    // Add a subflow
    diagram.addSubGraph(100, 100);
    const subflowNode = diagram.nodes.find((n) => n.type === "subflow")!;
    expect(subflowNode.data.isCollapsed).toBe(false);

    // Manually add a child inside the subflow
    const stereo = getLinearStereotype(diagram);
    diagram.addModule(stereo, 0, 0);
    const linearNode = diagram.nodes.find(
      (n) => n.data.stereotype === "Linear",
    )!;
    // Set parentId by direct mutation (no undo capture)
    diagram.nodes = diagram.nodes.map((n) =>
      n.id === linearNode.id ? { ...n, parentId: subflowNode.id } : n,
    );

    // Toggle to collapse
    diagram.toggleSubflow(subflowNode.id, true);
    const collapsedNode = diagram.nodes.find((n) => n.id === subflowNode.id)!;
    expect(collapsedNode.data.isCollapsed).toBe(true);

    // Undo should restore expanded state
    const ok = diagram.undo();
    expect(ok).toBe(true);
    const restoredNode = diagram.nodes.find((n) => n.id === subflowNode.id)!;
    expect(restoredNode.data.isCollapsed).toBe(false);
  });
});
