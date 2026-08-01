/*
 * Validate selected source diagrams before the Python pipeline runs.
 *
 * Targets:
 *  - NNM_MODEL_PATH (explicit selection), or
 *  - the manifest validation models (mninst, autoencoder_mnist) — the models
 *    that declare `outputShape` + `refreshTypesClean` in examples/manifest.json.
 *
 * Usage:
 *   NNM_MODEL_PATH=examples/diagrams/mninst.json \
 *     pnpm test:integration -- model-validation
 */

import { describe, expect, it, afterAll } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getModelPath,
  getValidationModels,
  DIAGRAMS_DIR,
  assertNNTreeReferenceIntegrity,
  parseNNTree,
  type NamedEntry,
} from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

const manifest = loadManifest();

/** Import a source diagram from disk and return the Diagram instance. */
function importDiagram(filePath: string): Diagram {
  const diagram = new Diagram();
  diagram.importFromJson(readFileSync(filePath, "utf-8"));
  return diagram;
}

describe("model validation", () => {
  const explicitModel = process.env.NNM_MODEL_PATH;

  if (explicitModel) {
    // Explicit selection (single model, no manifest filtering).
    const modelPath = getModelPath();

    it("loads the configured non-converted diagram", () => {
      expect(modelPath).not.toContain("/nntrees/");
      expect(modelPath).toMatch(/\.json$/);

      const diagram = importDiagram(modelPath);
      expect(diagram.nodes.length).toBeGreaterThan(0);
    });

    it("has no hard type errors before conversion", () => {
      const result = importDiagram(modelPath).refreshTypes();
      expect(result.errors.filter((error) => error.severity === "error")).toEqual([]);
    });

    it("compiles the selected source diagram to an NNTree", () => {
      const diagram = importDiagram(modelPath);
      const tree = parseNNTree(new NNTree(diagram).toJson());

      expect(tree.root).toBeTruthy();
      expect(Object.keys(tree.nodes)).not.toHaveLength(0);
      expect(tree.lossNode).toBeTruthy();
      assertNNTreeReferenceIntegrity(tree);
    });
  } else {
    // Manifest-driven: the validation models (mninst, autoencoder_mnist).
    const validationModels = getValidationModels(manifest);

    it("the manifest declares validation models", () => {
      expect(validationModels.length).toBeGreaterThan(0);
    });

    describe.each(validationModels)("$name", ({ name, entry }: NamedEntry) => {
      const filePath = resolve(DIAGRAMS_DIR, `${name}.json`);

      it("imports the source diagram", () => {
        const diagram = importDiagram(filePath);
        expect(diagram.nodes.length).toBeGreaterThan(0);
      });

      it("has no hard type errors before conversion", () => {
        const result = importDiagram(filePath).refreshTypes();
        expect(result.errors.filter((error) => error.severity === "error")).toEqual([]);
      });

      it("compiles to an NNTree with valid references", () => {
        const diagram = importDiagram(filePath);
        const tree = parseNNTree(new NNTree(diagram).toJson());

        expect(tree.root).toBeTruthy();
        expect(Object.keys(tree.nodes)).not.toHaveLength(0);
        expect(tree.lossNode).toBeTruthy();
        expect((tree.lossNode as { taskType?: string }).taskType).toBe(entry.taskType);
        assertNNTreeReferenceIntegrity(tree);
      });
    });
  }
});
