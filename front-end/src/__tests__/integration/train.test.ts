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
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getTargetDiagrams,
  tempDir,
  conditionalCleanup,
  runConvert,
  runTraining,
  composeHydraConfig,
  compileDiagramToFile,
  requirePython,
  CONVERTED_DIR,
  uvRun,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "train";

const PYTHON_AVAILABLE = requirePython();

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Python script verifying the reload invariants on a trained output dir:
 *  1. weights.safetensors is loadable via the safe loader;
 *  2. its keys match the freshly-built model's state_dict;
 *  3. eval() output is identical before and after reload on a fixed input.
 */
function generateReloadScript(convertedSrcPath: string): string {
  return `
import sys
import json
from hydra import compose, initialize_config_dir
import torch
from safetensors.torch import load_file

sys.path.insert(0, ${JSON.stringify(convertedSrcPath)})
from net.base import Net

cfg_dir = sys.argv[1]
safe_weights = sys.argv[2]
full_weights = sys.argv[3]

with initialize_config_dir(config_dir=cfg_dir):
    cfg = compose(config_name="base")

# 1. Safe weights must load with the safetensors loader.
safe_state = load_file(safe_weights)

# 2. Key set must match a freshly instantiated model.
net = Net(cfg)
state_keys = set(net.state_dict().keys())
safe_keys = set(safe_state.keys())
key_match = state_keys == safe_keys

# 3. Eval equivalence before/after reload on a fixed deterministic input.
net.load_state_dict(safe_state)
net.eval()
torch.manual_seed(0)
x = torch.randn(2, 1, 28, 28)
with torch.no_grad():
    out_before = net(x)

reloaded = torch.load(full_weights, map_location="cpu", weights_only=False)
reloaded.eval()
with torch.no_grad():
    out_after = reloaded(x)

result = {
    "safe_keys": len(safe_keys),
    "key_match": bool(key_match),
    "eval_equal": bool(torch.equal(out_before, out_after)),
    "success": True,
}
print(json.dumps(result))
`;
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
            const jsonPath = compileDiagramToFile(name, workDir);

            let cfgDir: string;
            try {
              cfgDir = runConvert(jsonPath, {
                numClasses: entry.numClasses,
                dataset: entry.dataset,
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
            const expectedDataset =
              entry.dataset ?? "dataset.mnist.MNISTDataset";
            expect(composed.stdout).toContain(`_target_: ${expectedDataset}`);

            // Transformer scope is not expanded in this milestone; the
            // default matrix now intentionally includes the autoencoder.
            if (!process.env.NNM_DIAGRAM) {
              expect(name).not.toMatch(/transformer/i);
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

            // Both weight artifacts are produced by main.py: the pickled full
            // model and the safetensors state dict.
            const safeWeightsPath = join(trainResult.outputDir, "weights.safetensors");
            expect(existsSync(safeWeightsPath)).toBe(true);
          },
        );

        it(
          "safe weights are loadable and eval output is reload-stable",
          { timeout: 600_000 },
          () => {
            // Reuses the same compile+convert+train path, then validates the
            // reload invariants on the artifacts produced by this run.
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileDiagramToFile(name, workDir);
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
              dataset: entry.dataset,
            });

            const trainResult = runTraining(cfgDir, {
              maxEpochs: 1,
              fastDevRun: process.env.NNM_FAST_DEV_RUN === "true",
              device: process.env.NNM_DEVICE || "cpu",
            });
            if (trainResult.exitCode !== 0) {
              throw new Error(
                `Training failed for ${name} (exit ${trainResult.exitCode}):\n` +
                  `${trainResult.stderr}\n${trainResult.stdout}`,
              );
            }

            const safeWeightsPath = join(trainResult.outputDir, "weights.safetensors");
            expect(existsSync(safeWeightsPath)).toBe(true);
            expect(existsSync(trainResult.weightsPath)).toBe(true);

            const scriptPath = join(workDir, "reload.py");
            writeFileSync(
              scriptPath,
              generateReloadScript(resolve(CONVERTED_DIR, "src")),
              "utf-8",
            );

            const result = uvRun(
              [
                "python",
                scriptPath,
                cfgDir,
                safeWeightsPath,
                trainResult.weightsPath,
              ],
              { timeout: 300_000 },
            );

            if (result.exitCode !== 0) {
              throw new Error(
                `Reload validation failed for ${name} (exit ${result.exitCode}):\n` +
                  `STDERR:\n${result.stderr}\nSTDOUT:\n${result.stdout}`,
              );
            }

            const output = JSON.parse(result.stdout.trim());
            expect(output.success).toBe(true);
            expect(output.safe_keys).toBeGreaterThan(0);
            expect(output.key_match).toBe(true);
            expect(output.eval_equal).toBe(true);
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
