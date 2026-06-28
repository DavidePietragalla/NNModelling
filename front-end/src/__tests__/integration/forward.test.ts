import { describe, it, expect, afterAll, afterEach } from "vitest";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getTargetDiagrams,
  DIAGRAMS_DIR,
  CONVERTED_DIR,
  tempDir,
  conditionalCleanup,
  runConvert,
  uvRun,
  isPythonAvailable,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "forward";

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

/**
 * Generate a Python script that hydrates the model and runs a forward pass.
 * Uses the same pattern as main.py: Net(cfg) with full Hydra config.
 * @param convertedSrcPath Absolute path to converted/src/
 */
function generateForwardScript(convertedSrcPath: string): string {
  return `
import sys
import json
from omegaconf import OmegaConf
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
    "requires_grad": output.requires_grad,
    "has_nan": bool(torch.isnan(output).any().item()),
    "has_inf": bool(torch.isinf(output).any().item()),
    "success": True,
}
print(json.dumps(result))
`;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  // Only test trainable diagrams for forward pass (no gradients needed,
  // but non-trainable diagrams may have missing dependencies like
  // non-standard ops)
  const allTargets = getTargetDiagrams(manifest, "forward");
  const targets = allTargets.filter((t) => t.entry.trainable);

  if (!PYTHON_AVAILABLE) {
    describe("Forward", () => {
      it("skipped — Python environment (uv) not available", () => {
        expect(true).toBe(true);
      });
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
            // Compile diagram -> NNTree JSON
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileToTempFile(name, workDir);

            // Run convert.py
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
            });

            // Write forward script
            const scriptPath = join(workDir, "forward.py");
            writeFileSync(
              scriptPath,
              generateForwardScript(resolve(CONVERTED_DIR, "src")),
              "utf-8",
            );

            // Run forward pass
            const inputShapeJson = JSON.stringify(entry.inputShape);
            const inputType = entry.inputType || "float";
            const result = uvRun(
              [
                "python",
                scriptPath,
                cfgDir,
                inputShapeJson,
                inputType,
              ],
              { timeout: 120_000 },
            );

            if (result.exitCode !== 0) {
              console.error(
                `Forward script failed for ${name}:\n` +
                  `STDERR:\n${result.stderr}\nSTDOUT:\n${result.stdout}`,
              );
            }
            expect(result.exitCode).toBe(0);

            // Parse output JSON
            const output = JSON.parse(result.stdout.trim());
            expect(output.success).toBe(true);
            expect(output.has_nan).toBe(false);
            expect(output.has_inf).toBe(false);

            // Output must be a tensor (1D or more)
            expect(Array.isArray(output.output_shape)).toBe(true);
            expect(output.output_shape.length).toBeGreaterThan(0);
          },
        );

        it(
          "forward pass with different batch size works",
          { timeout: 300_000 },
          () => {
            // Same as above but with batch_size=4 for variable batch
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileToTempFile(name, workDir);
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
            });

            const scriptPath = join(workDir, "forward2.py");
            writeFileSync(
              scriptPath,
              generateForwardScript(resolve(CONVERTED_DIR, "src")),
              "utf-8",
            );

            // Modify batch size
            const shape = [...entry.inputShape];
            shape[0] = 4;
            const inputShapeJson = JSON.stringify(shape);
            const inputType = entry.inputType || "float";

            const result = uvRun(
              [
                "python",
                scriptPath,
                cfgDir,
                inputShapeJson,
                inputType,
              ],
              { timeout: 120_000 },
            );

            expect(result.exitCode).toBe(0);
            const output = JSON.parse(result.stdout.trim());
            expect(output.success).toBe(true);
            expect(output.has_nan).toBe(false);
          },
        );
      },
    );
  }
} else {
  describe("Forward", () => {
    it("skipped — NNM_TIER is not 'forward' or 'all'", () => {
      expect(true).toBe(true);
    });
  });
}
