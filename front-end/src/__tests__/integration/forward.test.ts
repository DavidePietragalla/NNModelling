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
import { writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getTargetDiagrams,
  CONVERTED_DIR,
  tempDir,
  conditionalCleanup,
  runConvert,
  compileDiagramToFile,
  requirePython,
  uvRun,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "forward";

const PYTHON_AVAILABLE = requirePython();

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Generate a Python script that hydrates the model and runs a forward pass.
 * Uses the same pattern as main.py: Net(cfg) with full Hydra config.
 * @param convertedSrcPath Absolute path to converted/src/
 */
function generateForwardScript(convertedSrcPath: string): string {
  return `
import sys
import json
from hydra import compose, initialize_config_dir
import torch

# Add converted/src to the Python path
sys.path.insert(0, ${JSON.stringify(convertedSrcPath)})
from net.base import Net

cfg_dir = sys.argv[1]
input_shape = json.loads(sys.argv[2])
input_type = sys.argv[3] if len(sys.argv) > 3 else "float"

# Load the full Hydra config (same pattern as main.py)
with initialize_config_dir(config_dir=cfg_dir):
    cfg = compose(config_name="base")

# Instantiate model using Net(cfg) — same as main.py
net = Net(cfg)
net.eval()

# Create synthetic input
if input_type == "int":
    x = torch.randint(0, 100, input_shape, dtype=torch.long)
else:
    x = torch.randn(*input_shape)

# Forward pass
with torch.no_grad():
    output = net(x)

result = {
    "output_shape": list(output.shape),
    "output_dtype": str(output.dtype),
    "requires_grad": output.requires_grad,
    "has_nan": bool(torch.isnan(output).any().item()),
    "has_inf": bool(torch.isinf(output).any().item()),
    "success": True,
}
print(json.dumps(result))
`;
}

/** Run a single forward pass against a compiled+converted diagram. */
function runForwardPass(
  name: string,
  entry: NamedEntry["entry"],
  workDir: string,
  shape: number[],
): { output_shape: number[]; output_dtype: string; has_nan: boolean; has_inf: boolean } {
  const jsonPath = compileDiagramToFile(name, workDir);
  const cfgDir = runConvert(jsonPath, {
    numClasses: entry.numClasses,
    dataset: entry.dataset,
  });

  const scriptPath = join(workDir, "forward.py");
  writeFileSync(
    scriptPath,
    generateForwardScript(resolve(CONVERTED_DIR, "src")),
    "utf-8",
  );

  const inputShapeJson = JSON.stringify(shape);
  const inputType = entry.inputType || "float";
  const result = uvRun(
    ["python", scriptPath, cfgDir, inputShapeJson, inputType],
    { timeout: 120_000 },
  );

  if (result.exitCode !== 0) {
    throw new Error(
      `Forward script failed for ${name} (exit ${result.exitCode}):\n` +
        `STDERR:\n${result.stderr}\nSTDOUT:\n${result.stdout}`,
    );
  }

  const output = JSON.parse(result.stdout.trim());
  if (!output.success) {
    throw new Error(`Forward script reported failure for ${name}`);
  }
  return output;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  // Only test trainable diagrams for forward pass (non-trainable diagrams may
  // have missing dependencies like non-standard ops).
  const allTargets = getTargetDiagrams(manifest, "forward");
  const targets = allTargets.filter((t) => t.entry.trainable);

  if (!PYTHON_AVAILABLE) {
    describe.skip("Forward — Python environment unavailable", () => {
      it("requires uv and the converted Python environment", () => {});
    });
  } else {
    describe.each(targets)(
      "Forward: $name",
      ({ name, entry }: NamedEntry) => {
        const tmpDirs: string[] = [];

        afterEach(() => {
          for (const d of tmpDirs) {
            conditionalCleanup(d);
          }
        });

        it(
          "forward pass produces correct output shape",
          { timeout: 300_000 },
          () => {
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const output = runForwardPass(name, entry, workDir, entry.inputShape);

            expect(Array.isArray(output.output_shape)).toBe(true);
            expect(output.output_shape.length).toBeGreaterThan(0);
            expect(output.has_nan).toBe(false);
            expect(output.has_inf).toBe(false);

            // Exact shape/dtype contract is only asserted for models that
            // declare it (mninst and autoencoder_mnist).
            if (entry.outputShape) {
              const expectedShape = [entry.inputShape[0], ...entry.outputShape];
              expect(output.output_shape).toEqual(expectedShape);
              if (entry.outputDtype) {
                expect(output.output_dtype).toBe(`torch.${entry.outputDtype}`);
              }
            }
          },
        );

        it(
          "forward pass with different batch size works",
          { timeout: 300_000 },
          () => {
            const workDir = tempDir();
            tmpDirs.push(workDir);

            const shape = [...entry.inputShape];
            shape[0] = 4;
            const output = runForwardPass(name, entry, workDir, shape);

            expect(output.has_nan).toBe(false);
            expect(output.has_inf).toBe(false);

            if (entry.outputShape) {
              const expectedShape = [4, ...entry.outputShape];
              expect(output.output_shape).toEqual(expectedShape);
            }
          },
        );
      },
    );
  }
} else {
  describe.skip("Forward tier disabled", () => {
    it("runs only when NNM_TIER is forward or all", () => {});
  });
}
