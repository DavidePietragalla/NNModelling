/*
 * Validate a user-selectable source diagram before the Python pipeline runs.
 *
 * Usage:
 *   NNM_MODEL_PATH=examples/diagrams/mninst.json \
 *     pnpm test:integration -- model-validation
 */

import { describe, expect, it, afterAll } from "vitest";
import { readFileSync } from "node:fs";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow } from "../helpers";
import { getModelPath } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

describe("selected source model", () => {
  it("loads the configured non-converted diagram", () => {
    const modelPath = getModelPath();
    expect(modelPath).not.toContain("/nntrees/");
    expect(modelPath).toMatch(/\.json$/);

    const diagram = new Diagram();
    diagram.importFromJson(readFileSync(modelPath, "utf-8"));

    expect(diagram.nodes.length).toBeGreaterThan(0);
  });

  it("has no hard type errors before conversion", () => {
    const diagram = new Diagram();
    diagram.importFromJson(readFileSync(getModelPath(), "utf-8"));

    const result = diagram.refreshTypes();
    expect(result.errors.filter((error) => error.severity === "error")).toEqual([]);
  });

  it("compiles the selected source diagram to an NNTree", () => {
    const diagram = new Diagram();
    diagram.importFromJson(readFileSync(getModelPath(), "utf-8"));
    const tree = JSON.parse(new NNTree(diagram).toJson()) as {
      root: string;
      nodes: Record<string, unknown>;
      lossNode: unknown;
    };

    expect(tree.root).toBeTruthy();
    expect(Object.keys(tree.nodes)).not.toHaveLength(0);
    expect(tree.lossNode).toBeTruthy();
  });
});
