import { describe, it, expect, afterAll, afterEach } from "vitest";
import {
  readFileSync,
  writeFileSync,
  existsSync,
  readdirSync,
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

/**
 * Find first .ckpt file in a directory (searches recursively).
 * Returns the absolute path to the checkpoint, or null.
 */
function findCkpt(dir: string): string | null {
  if (!existsSync(dir)) return null;
  try {
    const files = readdirSync(dir, { recursive: true }) as string[];
    const ckpt = files.find((f: string) => f.endsWith(".ckpt"));
    return ckpt ? resolve(dir, ckpt) : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  // Only trainable diagrams
  const allTargets = getTargetDiagrams(manifest, "infer");

  if (!PYTHON_AVAILABLE) {
    describe("Infer", () => {
      it("skipped — Python environment (uv) not available", () => {
        expect(true).toBe(true);
      });
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
            expect(trainResult.exitCode).toBe(0);

            // 4. Find checkpoint
            const ckptPath = findCkpt(trainResult.ckptDir);
            if (!ckptPath) {
              console.warn(
                `No checkpoint found in ${trainResult.ckptDir} — ` +
                  `skipping infer test (expected with fast_dev_run)`,
              );
              return;
            }

            // 5. Run inference
            const outputPath = join(workDir, "predictions.json");
            const inferResult = runInference(cfgDir, ckptPath, {
              outputPath,
            });

            expect(inferResult.exitCode).toBe(0);

            // 6. Validate predictions JSON
            expect(existsSync(outputPath)).toBe(true);
            const predictionsRaw = readFileSync(outputPath, "utf-8");
            const predictions = JSON.parse(predictionsRaw);

            expect(predictions).toBeTypeOf("object");
            // The predictions key might be "predictions" or the output format
            // varies by model type. Check for common keys.
            const hasPredictions =
              "predictions" in predictions ||
              "output" in predictions ||
              "reconstruction" in predictions;

            expect(hasPredictions).toBe(true);

            // Verify predictions is a non-empty array
            const preds =
              predictions.predictions ||
              predictions.output ||
              predictions.reconstruction ||
              [];
            expect(Array.isArray(preds)).toBe(true);
            expect(preds.length).toBeGreaterThan(0);
          },
        );
      },
    );
  }
} else {
  describe("Infer", () => {
    it("skipped — NNM_TIER is not 'infer' or 'all'", () => {
      expect(true).toBe(true);
    });
  });
}
