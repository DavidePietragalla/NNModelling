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
import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
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
  composeHydraConfig,
  isPythonAvailable,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "train";

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
  // Only trainable diagrams, further filtered by env vars
  const allTargets = getTargetDiagrams(manifest, "train");

  if (!PYTHON_AVAILABLE) {
    describe.skip("Train — Python environment unavailable", () => {
      it("requires uv and the converted Python environment", () => {});
    });
  } else {
    describe.each(allTargets)(
      "Train: $name",
      ({ name, entry }: NamedEntry) => {
        const tmpDirs: string[] = [];

        afterEach(() => {
          for (const d of tmpDirs) {
            conditionalCleanup(d);
          }
        });

        it(
          "converts and trains for 1 epoch",
          { timeout: 600_000 },
          () => {
            // Tier 1: Compile diagram -> NNTree JSON -> convert.py
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileToTempFile(name, workDir);

            let cfgDir: string;
            try {
              cfgDir = runConvert(jsonPath, {
                numClasses: entry.numClasses,
              });
            } catch (e: unknown) {
              // If convert.py fails, mark as skipped with explanation
              const msg = e instanceof Error ? e.message : String(e);
              console.warn(`Convert failed for ${name}: ${msg}`);
              throw e;
            }

            // Prove that Hydra can compose this exact generated directory
            // before spending time on training.
            const netYaml = readFileSync(
              join(cfgDir, "net", "custom_sequence.yaml"),
              "utf-8",
            );
            const root = /^root:\s*([^\s]+)$/m.exec(netYaml)?.[1];
            expect(root).toBeDefined();
            expect(netYaml).toContain("nodes:");

            const composed = composeHydraConfig(cfgDir);
            if (composed.exitCode !== 0) {
              throw new Error(
                `Hydra composition failed for ${name} (exit ${composed.exitCode}, signal ${composed.signal ?? "none"}):\n` +
                  `${composed.error ?? ""}\nSTDERR:\n${composed.stderr}\nSTDOUT:\n${composed.stdout}`,
              );
            }
            expect(composed.stdout).toContain(`root: ${root}`);
            expect(composed.stdout).toContain("_target_: torch.nn.Linear");
            expect(composed.stdout).toContain("_target_: dataset.mnist.MNISTDataset");

            if (!process.env.NNM_DIAGRAM) {
              expect(name).not.toMatch(/transformer|auto.?encoder/i);
            }

            // Tier 3: Train for 1 epoch
            const startedAt = Date.now();
            const trainResult = runTraining(cfgDir, {
              maxEpochs: 1,
              fastDevRun: process.env.NNM_FAST_DEV_RUN === "true",
              device: process.env.NNM_DEVICE || "cpu",
            });

            if (trainResult.exitCode !== 0) {
              throw new Error(
                `Training failed for ${name} (exit ${trainResult.exitCode}, signal ${trainResult.signal ?? "none"}):\n` +
                  `${trainResult.error ?? ""}\nSTDERR:\n${trainResult.stderr}\nSTDOUT:\n${trainResult.stdout}`,
              );
            }

            expect(trainResult.stdout).toContain("Training...");
            expect(existsSync(trainResult.weightsPath)).toBe(true);
            expect(statSync(trainResult.weightsPath).mtimeMs).toBeGreaterThanOrEqual(startedAt);
          },
        );
      },
    );
  }
} else {
  describe.skip("Train tier disabled", () => {
    it("runs only when NNM_TIER is train or all", () => {});
  });
}
