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

import { describe, it, expect, afterAll, afterEach } from "vitest";
import {
  readFileSync,
  writeFileSync,
  existsSync,
} from "node:fs";
import { resolve, join } from "node:path";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getTargetDiagrams,
  DIAGRAMS_DIR,
  tempDir,
  conditionalCleanup,
  runConvert,
  runTraining,
  runInference,
  isPythonAvailable,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "infer";

const PYTHON_AVAILABLE = isPythonAvailable();

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function compileToTempFile(name: string, outDir: string): string {
  const diagramPath = resolve(DIAGRAMS_DIR, `${name}.json`);
  const content = readFileSync(diagramPath, "utf-8");

  const diagram = new Diagram();
  diagram.importFromJson(content);

  const nnTree = new NNTree(diagram);
  const jsonStr = nnTree.toJson();

  const jsonPath = join(outDir, `${name}.json`);
  writeFileSync(jsonPath, jsonStr, "utf-8");
  return jsonPath;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  // Only trainable diagrams
  const allTargets = getTargetDiagrams(manifest, "infer");

  if (!PYTHON_AVAILABLE) {
    describe.skip("Infer — Python environment unavailable", () => {
      it("requires uv and the converted Python environment", () => {});
    });
  } else {
    describe.each(allTargets)(
      "Infer: $name",
      ({ name, entry }: NamedEntry) => {
        const tmpDirs: string[] = [];

        afterEach(() => {
          for (const d of tmpDirs) {
            conditionalCleanup(d);
          }
        });

        it(
          "full pipeline: compile -> convert -> train -> infer",
          { timeout: 900_000 },
          () => {
            // 1. Compile diagram -> NNTree JSON
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileToTempFile(name, workDir);

            // 2. Convert
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
            });

            // 3. Train
            const trainResult = runTraining(cfgDir, {
              maxEpochs: 1,
              fastDevRun: process.env.NNM_FAST_DEV_RUN === "true",
              device: process.env.NNM_DEVICE || "cpu",
            });
            if (trainResult.exitCode !== 0) {
              throw new Error(
                `Training failed for ${name}:\n${trainResult.stderr}\n${trainResult.stdout}`,
              );
            }
            expect(existsSync(trainResult.weightsPath)).toBe(true);

            // 4. Run inference with the model artifact created by this test.
            const outputPath = join(workDir, "predictions.json");
            const inferResult = runInference(cfgDir, trainResult.weightsPath, {
              outputPath,
            });

            if (inferResult.exitCode !== 0) {
              throw new Error(
                `Inference failed for ${name}:\n${inferResult.stderr}\n${inferResult.stdout}`,
              );
            }

            // 6. Validate predictions JSON
            expect(existsSync(outputPath)).toBe(true);
            const predictionsRaw = readFileSync(outputPath, "utf-8");
            const predictions = JSON.parse(predictionsRaw);

            expect(Array.isArray(predictions)).toBe(true);
            expect(predictions.length).toBeGreaterThan(0);
            expect(predictions[0]).toHaveProperty("prediction");
          },
        );
      },
    );
  }
} else {
  describe.skip("Infer tier disabled", () => {
    it("runs only when NNM_TIER is infer or all", () => {});
  });
}
