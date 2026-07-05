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

/** Extract params from a node's data, normalizing to { value, position } shape. */
function getNodeParams(diagram: Diagram, nodeId: string): Record<string, { value: string; position?: string }> {
  const node = diagram.getNodeById(nodeId);
  if (!node) throw new Error(`Node "${nodeId}" not found`);
  return (node.data.params as Record<string, { value: string; position?: string }>) ?? {};
}

// ── Params Merge Tests ─────────────────────────────────────────────────
// BUG: DiagramCore.addModule and addJoinNode use all-or-nothing for params.
// When user provides ANY params, stereotype defaults are discarded entirely.
// Expected: user params should OVERLAY stereotype defaults.
// These tests should FAIL until the fix is applied.

describe("Params merge in addModule", () => {
  it("BUG: preserves stereotype defaults when user provides partial params", () => {
    const diagram = new Diagram();
    // Auto-spawned Input node is ignored
    const linear = getStereotype(diagram, "Linear");

    // Linear stereotype has: in_features (Undefined), out_features (Undefined),
    // bias (True), device (None), dtype (None)
    diagram.addModule(linear, 100, 100, {
      name: "MyLinear",
      params: { in_features: "128", out_features: "64" },
    });

    const added = diagram.nodes[diagram.nodes.length - 1]; // last node added
    const params = getNodeParams(diagram, added.id);

    // User overrides should replace defaults for specified keys
    expect(params.in_features).toEqual({ value: "128", position: "top" });
    expect(params.out_features).toEqual({ value: "64", position: "bottom" });

    // Stereotype defaults SHOULD be preserved for keys not in user params
    // THIS IS THE BUG: these will be undefined or empty because addModule
    // does not merge with stereotype defaults.
    expect(params.bias).toBeDefined();
    expect(params.bias?.value).toBe("True");
    expect(params.device).toBeDefined();
    expect(params.device?.value).toBe("None");
    expect(params.dtype).toBeDefined();
    expect(params.dtype?.value).toBe("None");
  });

  it("BUG: all stereotype defaults present when no user params provided", () => {
    const diagram = new Diagram();
    const linear = getStereotype(diagram, "Linear");

    diagram.addModule(linear, 100, 100, { name: "NoParams" });
    const added = diagram.nodes[diagram.nodes.length - 1];
    const params = getNodeParams(diagram, added.id);

    // Without user params, all defaults should be used and wrapped in { value, position }
    // THIS IS ALSO A BUG: currently returns {} instead of stereotype defaults
    expect(params.in_features).toBeDefined();
    expect(params.in_features?.value).toBe("Undefined");
    expect(params.in_features?.position).toBe("top");
    expect(params.out_features).toBeDefined();
    expect(params.out_features?.value).toBe("Undefined");
    expect(params.out_features?.position).toBe("bottom");
    expect(params.bias).toBeDefined();
    expect(params.bias?.value).toBe("True");
  });

  it("BUG: user params wrapped in { value } format match set_parameter convention", () => {
    // set_parameter uses { value: string } wrapper. create_node params should too.
    const diagram = new Diagram();
    const linear = getStereotype(diagram, "Linear");

    diagram.addModule(linear, 100, 100, {
      params: { in_features: "256" },
    });

    const added = diagram.nodes[diagram.nodes.length - 1];
    const params = getNodeParams(diagram, added.id);

    // User values should be wrapped in { value } format (same as set_parameter)
    // BUG: currently stored as plain string "256", not { value: "256" }
    const param = params.in_features;
    expect(typeof param).toBe("object");
    expect(param).toHaveProperty("value");
    expect(param.value).toBe("256");
  });

  it("BUG: Tanh node with no stereotype params gets empty params (not undefined)", () => {
    const diagram = new Diagram();
    const tanh = getStereotype(diagram, "Tanh");

    diagram.addModule(tanh, 100, 100, { name: "MyTanh" });
    const added = diagram.nodes[diagram.nodes.length - 1];

    // Tanh has no parameters — should get an empty object, not undefined
    expect(added.data.params).toBeDefined();
    expect(added.data.params).toEqual({});
  });
});

describe("Params merge in addJoinNode", () => {
  it("join node with user params wraps them in { value } format", () => {
    const diagram = new Diagram();
    const concat = getStereotype(diagram, "Concat");

    // Concat stereotype has: dim (default "-1", position "top")
    diagram.addJoinNode(concat, 100, 100, {
      params: { dim: "1" },
    });

    const added = diagram.nodes[diagram.nodes.length - 1];
    const params = getNodeParams(diagram, added.id);

    // User value should be wrapped in { value, position } format
    expect(params.dim).toBeDefined();
    expect(typeof params.dim).toBe("object");
    expect(params.dim.value).toBe("1");
    // Position from stereotype should be preserved
    expect(params.dim.position).toBe("top");
  });

  it("join node with no user params uses stereotype defaults", () => {
    const diagram = new Diagram();
    const concat = getStereotype(diagram, "Concat");

    diagram.addJoinNode(concat, 100, 100);
    const added = diagram.nodes[diagram.nodes.length - 1];
    const params = getNodeParams(diagram, added.id);

    // Should use the stereotype default
    expect(params.dim).toBeDefined();
    expect(params.dim.value).toBe("-1");
    expect(params.dim.position).toBe("top");
  });
});
