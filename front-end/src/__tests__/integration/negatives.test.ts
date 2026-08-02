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
 *
 * Negative and contract tests. The input-cardinality contracts (no-Input /
 * duplicate-Input diagrams are rejected) and the D1/D7 decisions: editor
 * validation and NNTree compilation both reject directed cycles, while a
 * valid DAG reconvergence on a join compiles without a false loop warning.
 */

import { describe, it, expect, afterAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow } from "../helpers";
import { DIAGRAMS_DIR, parseNNTree } from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const tier = (process.env.NNM_TIER || "all") as string;
const shouldRun = tier === "all" || tier === "smoke";

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function linearNode(id: string): Record<string, unknown> {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      stereotype: "Linear",
      name: `Linear_${id}`,
      color: "#4779c4",
      params: {
        in_features: { value: "784" },
        out_features: { value: "10" },
      },
      isInput: false,
      isLoss: false,
    },
  };
}

function inputNode(id: string): Record<string, unknown> {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      stereotype: "Input",
      name: `Input_${id}`,
      color: "#27b376",
      params: {},
      isInput: true,
      isLoss: false,
    },
  };
}

function importDiagram(name: string): Diagram {
  const diagram = new Diagram();
  diagram.importFromJson(
    readFileSync(resolve(DIAGRAMS_DIR, `${name}.json`), "utf-8"),
  );
  return diagram;
}

/** Compile a diagram and return any captured console.warn output. */
function compileWithWarnings(diagram: Diagram): { warnings: string[]; tree: string } {
  const captured: string[] = [];
  const original = console.warn;
  console.warn = (...args: unknown[]) => captured.push(args.join(" "));
  try {
    const tree = new NNTree(diagram).toJson();
    return { warnings: captured, tree };
  } finally {
    console.warn = original;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  describe("Negative: input cardinality contract", () => {
    it("rejects a diagram with no Input node", () => {
      const diagram = new Diagram();
      const json = JSON.stringify({
        nodes: [linearNode("l1")],
        edges: [],
      });
      diagram.importFromJson(json);

      expect(() => new NNTree(diagram)).toThrow(/exactly one input node/i);
    });

    it("rejects a diagram with two Input nodes", () => {
      const diagram = new Diagram();
      const json = JSON.stringify({
        nodes: [inputNode("in1"), inputNode("in2"), linearNode("l1")],
        edges: [{ id: "e1", source: "in1", target: "l1" }],
      });
      diagram.importFromJson(json);

      expect(() => new NNTree(diagram)).toThrow(/exactly one input node/i);
    });
  });

  describe("Negative: directed cycles are rejected (D1)", () => {
    it("NNTree compilation rejects a top-level directed cycle", () => {
      const diagram = new Diagram();
      const json = JSON.stringify({
        nodes: [inputNode("input"), linearNode("l1"), linearNode("l2")],
        edges: [
          { id: "e1", source: "input", target: "l1" },
          { id: "e2", source: "l1", target: "l2" },
          { id: "e3", source: "l2", target: "l1" },
        ],
      });
      diagram.importFromJson(json);

      expect(() => new NNTree(diagram)).toThrow(/cycle/i);
    });

    it("editor validation rejects an edge that would close a directed cycle", () => {
      const diagram = new Diagram();
      const json = JSON.stringify({
        nodes: [inputNode("input"), linearNode("l1"), linearNode("l2")],
        edges: [
          { id: "e1", source: "input", target: "l1" },
          { id: "e2", source: "l1", target: "l2" },
        ],
      });
      diagram.importFromJson(json);

      // Closing l2 -> l1 would create input -> l1 -> l2 -> l1.
      expect(() => diagram.addEdge("l2", "l1")).toThrow(/cycle/i);
    });
  });

  describe("Negative: DAG reconvergence on a join is not a cycle (D7)", () => {
    it("autoencoder_mnist compiles without the false skip_addition loop warning", () => {
      const diagram = importDiagram("autoencoder_mnist");
      const { warnings, tree } = compileWithWarnings(diagram);
      const parsed = parseNNTree(tree);

      expect(parsed.root).toBeTruthy();
      expect(parsed.lossNode).toBeTruthy();
      expect(
        warnings.some((w) => /visited|loop/i.test(w)),
        `unexpected loop warning: ${warnings.join(" | ")}`,
      ).toBe(false);
    });
  });
} else {
  describe.skip("Negative tier disabled", () => {
    it("runs only when NNM_TIER is smoke or all", () => {});
  });
}
