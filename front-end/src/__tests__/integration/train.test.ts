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
import { readFileSync, writeFileSync, readdirSync, existsSync } from "node:fs";
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
    describe("Train", () => {
      it("skipped — Python environment (uv) not available", () => {
        expect(true).toBe(true);
      });
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

            // Tier 3: Train for 1 epoch
            const trainResult = runTraining(cfgDir, {
              maxEpochs: 1,
              fastDevRun: process.env.NNM_FAST_DEV_RUN === "true",
              device: process.env.NNM_DEVICE || "cpu",
            });

            expect(trainResult.exitCode).toBe(0);

            // Check that stdout contains training log
            expect(trainResult.stdout.length).toBeGreaterThan(0);

            // Check for checkpoints — they may be in cfgDir or a subdirectory
            const hasCkptDir = existsSync(trainResult.ckptDir);
            if (hasCkptDir) {
              const allFiles = readdirSync(trainResult.ckptDir, {
                recursive: true,
              }) as string[];
              const ckpts = allFiles.filter(
                (f: string) => f.endsWith(".ckpt"),
              );
              // Note: With max_epochs=1 and fast_dev_run, there might
              // not be a checkpoint. This assertion is soft — we check
              // the exit code primarily.
              if (ckpts.length > 0) {
                console.log(
                  `Found ${ckpts.length} checkpoint(s) for ${name}`,
                );
              }
            }
          },
        );
      },
    );
  }
} else {
  describe("Train", () => {
    it("skipped — NNM_TIER is not 'train' or 'all'", () => {
      expect(true).toBe(true);
    });
  });
}
