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
  existsSync,
  createReadStream,
} from "node:fs";
import { resolve, join } from "node:path";
import { createInterface } from "node:readline";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getTargetDiagrams,
  tempDir,
  conditionalCleanup,
  runConvert,
  runTraining,
  runInference,
  compileDiagramToFile,
  requirePython,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "infer";

const PYTHON_AVAILABLE = requirePython();

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface Prediction {
  input: unknown;
  target: unknown;
  prediction: unknown;
}

function isNestedImageShape(value: unknown, shape: number[]): boolean {
  if (!Array.isArray(value)) return false;
  const [first, ...rest] = shape;
  if (rest.length === 0) {
    return (value as unknown[]).length === first;
  }
  return (
    (value as unknown[]).length === first &&
    (value as unknown[]).every((v) => isNestedImageShape(v, rest))
  );
}

/**
 * Count prediction records in the infer.py output without loading the whole
 * file: each record contributes exactly one `"prediction":` key line. The
 * autoencoder output can exceed 500 MB, so a full read would blow the Node
 * string limit.
 */
async function countPredictions(filePath: string): Promise<number> {
  const rl = createInterface({
    input: createReadStream(filePath, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  let count = 0;
  for await (const line of rl) {
    if (line.includes('"prediction":')) count++;
  }
  return count;
}

/**
 * Read up to maxRecords complete prediction objects from the start of the
 * file. The output contains only numeric JSON values, so brace balancing is
 * unambiguous and each record is small enough to parse individually.
 */
async function readPredictionSample(
  filePath: string,
  maxRecords: number,
): Promise<Prediction[]> {
  const stream = createReadStream(filePath, { encoding: "utf-8", highWaterMark: 64 * 1024 });
  const records: Prediction[] = [];
  let buffer = "";

  for await (const chunk of stream) {
    buffer += chunk;
    while (records.length < maxRecords) {
      const start = buffer.indexOf("{");
      if (start === -1) break;
      let depth = 0;
      let end = -1;
      for (let i = start; i < buffer.length; i++) {
        if (buffer[i] === "{") depth++;
        else if (buffer[i] === "}") {
          depth--;
          if (depth === 0) {
            end = i;
            break;
          }
        }
      }
      if (end === -1) break; // need more chunks for a complete record
      records.push(JSON.parse(buffer.slice(start, end + 1)) as Prediction);
      buffer = buffer.slice(end + 1);
    }
    if (records.length >= maxRecords) break;
  }

  if (records.length === 0) {
    throw new Error(`No prediction records found in ${filePath}`);
  }
  return records;
}

/** Validate the infer.py prediction JSON schema per task type. */
function assertPredictionSchema(
  predictions: Prediction[],
  entry: NamedEntry["entry"],
): void {
  expect(predictions.length).toBeGreaterThan(0);

  for (const record of predictions) {
    expect(record).toHaveProperty("input");
    expect(record).toHaveProperty("target");
    expect(record).toHaveProperty("prediction");

    // input is always the normalized image: nested [C, H, W] floats.
    expect(isNestedImageShape(record.input, [1, 28, 28])).toBe(true);

    if (entry.taskType === "classification") {
      const prediction = record.prediction;
      expect(typeof prediction).toBe("number");
      expect(Number.isInteger(prediction)).toBe(true);
      if (entry.numClasses !== undefined) {
        expect(prediction as number).toBeGreaterThanOrEqual(0);
        expect(prediction as number).toBeLessThan(entry.numClasses);
      }
      expect(typeof record.target).toBe("number");
    } else {
      // Regression/autoencoder: raw output with the declared per-sample shape.
      const expected = entry.outputShape ?? [1, 28, 28];
      expect(isNestedImageShape(record.prediction, expected)).toBe(true);
      expect(isNestedImageShape(record.target, [1, 28, 28])).toBe(true);
    }
  }
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
          async () => {
            // 1. Compile diagram -> NNTree JSON
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileDiagramToFile(name, workDir);

            // 2. Convert (with the manifest-declared dataset)
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
              dataset: entry.dataset,
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

            // 5. Validate predictions: schema, cardinality, nested shape, and
            // numeric/integer structure as applicable. JSON does not preserve
            // tensor dtype, so no dtype assertion is made here — exact dtype
            // is verified by the forward gate instead.
            // The autoencoder output file can exceed 500 MB, so both checks
            // are streaming / sample-based.
            expect(existsSync(outputPath)).toBe(true);
            const recordCount = await countPredictions(outputPath);
            expect(recordCount).toBeGreaterThan(0);

            // Cardinality matches the MNIST test set (both selected models
            // and the default train/infer matrix train on MNIST data).
            const dataset = entry.dataset ?? "dataset.mnist.MNISTDataset";
            if (dataset.includes("mnist")) {
              expect(recordCount).toBe(10_000);
            }

            const sample = await readPredictionSample(outputPath, 3);
            assertPredictionSchema(sample, entry);
          },
        );

        it(
          "inference rejects a missing weights file",
          { timeout: 300_000 },
          () => {
            const workDir = tempDir();
            tmpDirs.push(workDir);
            const jsonPath = compileDiagramToFile(name, workDir);
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
              dataset: entry.dataset,
            });

            const missingWeights = join(workDir, "does-not-exist.pt");
            const inferResult = runInference(cfgDir, missingWeights);

            // Contract: infer.py exits non-zero and reports the missing file.
            expect(inferResult.exitCode).not.toBe(0);
            expect(inferResult.stderr).toContain("Weights file not found");
          },
        );

        it.skipIf(!entry.supportsImages)(
          "image artifacts are produced when the dataset supports them",
          { timeout: 900_000 },
          () => {
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
                `Training failed for ${name}:\n${trainResult.stderr}\n${trainResult.stdout}`,
              );
            }

            const imageDir = join(workDir, "images");
            const inferResult = runInference(cfgDir, trainResult.weightsPath, {
              imageDir,
            });
            if (inferResult.exitCode !== 0) {
              throw new Error(
                `Inference (image) failed for ${name}:\n${inferResult.stderr}\n${inferResult.stdout}`,
              );
            }

            expect(existsSync(join(imageDir, "montage.png"))).toBe(true);
            expect(existsSync(join(imageDir, "sample_000.png"))).toBe(true);
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
