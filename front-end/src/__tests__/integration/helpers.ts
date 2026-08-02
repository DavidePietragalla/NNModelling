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

import { spawnSync } from "node:child_process";
import { resolve, dirname, join, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  mkdtempSync,
  rmSync,
  existsSync,
  readFileSync,
  mkdirSync,
  readdirSync,
  writeFileSync,
  symlinkSync,
  cpSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { expect } from "vitest";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const _filename = fileURLToPath(import.meta.url);
const _dirname = dirname(_filename);

/** NNModelling project root (parent of front-end/, converted/, examples/) */
export const PROJECT_ROOT = resolve(_dirname, "../../../..");

/** examples/ directory containing manifest.json, diagrams/, nntrees/ */
export const EXAMPLES_DIR = resolve(PROJECT_ROOT, "examples");

/** converted/ directory containing Python code */
export const CONVERTED_DIR = resolve(PROJECT_ROOT, "converted");

/** examples/diagrams/ — Svelte Flow format source diagrams */
export const DIAGRAMS_DIR = resolve(EXAMPLES_DIR, "diagrams");

/** examples/nntrees/ — pre-compiled NNTree JSON files */
export const NNTREES_DIR = resolve(EXAMPLES_DIR, "nntrees");

/**
 * Source model selected for model-validation integration tests.
 *
 * NNM_MODEL_PATH may be absolute or relative to the repository root. It must
 * point to a Svelte Flow source diagram, not an already compiled NNTree.
 */
export function getModelPath(): string {
  const configured = process.env.NNM_MODEL_PATH;
  const modelPath = configured
    ? resolve(PROJECT_ROOT, configured)
    : resolve(DIAGRAMS_DIR, `${process.env.NNM_DIAGRAM || "mninst"}.json`);
  if (!existsSync(modelPath)) {
    throw new Error(`Model path does not exist: ${modelPath}`);
  }
  if (modelPath.startsWith(`${NNTREES_DIR}/`)) {
    throw new Error(`NNM_MODEL_PATH must point to a source diagram, not an NNTree: ${modelPath}`);
  }
  return modelPath;
}

/** Manifest path */
export const MANIFEST_PATH = resolve(EXAMPLES_DIR, "manifest.json");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiagramEntry {
  format: "svelte-flow" | "nntree";
  inputShape: number[];
  inputType?: "float" | "int";
  /**
   * Expected model output shape for a single sample (batch dimension
   * excluded). Verified exactly by the forward tier. Only models that
   * declare this shape get the exact-shape assertion.
   */
  outputShape?: number[];
  /** Expected torch dtype of the model output, e.g. "float32". */
  outputDtype?: string;
  numClasses?: number;
  taskType: "classification" | "regression";
  /**
   * convert.py `--dataset` class path. Required for training/reload/inference
   * invariants (e.g. autoencoder diagrams must select AutoencoderMNIST or the
   * MSELoss target shape is wrong).
   */
  dataset?: string;
  trainable: boolean;
  /** Included in the default fast training/inference matrix. */
  trainingSmoke?: boolean;
  /**
   * The diagram must pass TypeEngine inference with no hard errors.
   * Diagrams with known pre-existing hard errors must not declare this.
   */
  refreshTypesClean?: boolean;
  /** infer.py --image-dir artifact path is validated for this model. */
  supportsImages?: boolean;
  description: string;
}

export interface Manifest {
  diagrams: Record<string, DiagramEntry>;
  nntrees: Record<string, DiagramEntry>;
}

export type Tier = "smoke" | "convert" | "forward" | "train" | "infer" | "all";

export interface SpawnResult {
  exitCode: number;
  stdout: string;
  stderr: string;
  signal?: NodeJS.Signals | null;
  error?: string;
}

export interface NamedEntry {
  name: string;
  entry: DiagramEntry;
}

export interface TrainResult {
  exitCode: number;
  outputDir: string;
  weightsPath: string;
  stdout: string;
  stderr: string;
  signal?: NodeJS.Signals | null;
  error?: string;
}

// ---------------------------------------------------------------------------
// Manifest helpers
// ---------------------------------------------------------------------------

/** Read and parse examples/manifest.json */
export function loadManifest(): Manifest {
  const raw = readFileSync(MANIFEST_PATH, "utf-8");
  return JSON.parse(raw) as Manifest;
}

/**
 * Get the list of diagrams to test based on NNM_DIAGRAM env var and tier.
 *
 * - When NNM_DIAGRAM is set: return only that specific diagram (or throw).
 * - For train/infer tiers: return only trainable diagrams.
 * - Otherwise: return all diagrams.
 */
export function getTargetDiagrams(
  manifest: Manifest,
  tier: Tier,
): Array<{ name: string; entry: DiagramEntry }> {
  const targetName = process.env.NNM_DIAGRAM || "";
  const diagrams = manifest.diagrams || {};

  if (targetName) {
    if (!diagrams[targetName]) {
      throw new Error(
        `Diagram "${targetName}" not found in manifest. ` +
          `Available: ${Object.keys(diagrams).join(", ")}`,
      );
    }
    return [{ name: targetName, entry: diagrams[targetName] }];
  }

  if (tier === "train" || tier === "infer") {
    return Object.entries(diagrams)
      .filter(([_, entry]) => entry.trainable && entry.trainingSmoke === true)
      .map(([name, entry]) => ({ name, entry }));
  }

  return Object.entries(diagrams).map(([name, entry]) => ({ name, entry }));
}

/**
 * Models that declare the full invariant set (exact output shape and a clean
 * type check). These are the models used for the strengthened invariants:
 * currently `mninst` and `autoencoder_mnist`.
 */
export function getValidationModels(
  manifest: Manifest,
): Array<{ name: string; entry: DiagramEntry }> {
  return Object.entries(manifest.diagrams)
    .filter(([_, entry]) => entry.outputShape !== undefined && entry.refreshTypesClean === true)
    .map(([name, entry]) => ({ name, entry }));
}

// ---------------------------------------------------------------------------
// NNTree helpers
// ---------------------------------------------------------------------------

export interface ParsedNNTree {
  root: string;
  lossNode: unknown;
  nodes: Record<string, unknown>;
}

interface ParsedNode {
  data?: { type?: string; [key: string]: unknown };
  children?: string[];
}

/** Parse an NNTree JSON string into its structural shape. */
export function parseNNTree(jsonStr: string): ParsedNNTree {
  return JSON.parse(jsonStr) as ParsedNNTree;
}

/**
 * Assert structural reference integrity of a compiled NNTree:
 * every declared child ID resolves to a tree-level node, the root exists, and
 * every subflow's internal graph references only internal or tree-level nodes.
 */
export function assertNNTreeReferenceIntegrity(tree: ParsedNNTree): void {
  const treeIds = new Set(Object.keys(tree.nodes));

  expect(
    treeIds.has(tree.root),
    `root "${tree.root}" must exist in the tree nodes map`,
  ).toBe(true);

  for (const [id, node] of Object.entries(tree.nodes)) {
    const parsed = node as ParsedNode;
    const data = parsed.data ?? {};
    for (const childId of parsed.children ?? []) {
      expect(
        treeIds.has(childId),
        `node ${id} references missing tree node ${childId}`,
      ).toBe(true);
    }
    if (data.type === "subflow") {
      const sfData = data as unknown as {
        entryNode?: string;
        nodes?: Record<string, { children?: string[] }>;
      };
      const internal = sfData.nodes ?? {};
      expect(
        sfData.entryNode !== undefined && sfData.entryNode in internal,
        `subflow ${id} entryNode "${sfData.entryNode}" must exist internally`,
      ).toBe(true);
      for (const [intId, intNode] of Object.entries(internal)) {
        for (const childId of intNode.children ?? []) {
          const isInternal = childId in internal;
          const isTreeLevel = treeIds.has(childId);
          expect(
            isInternal || isTreeLevel,
            `subflow ${id} internal node ${intId} references unknown node ${childId}`,
          ).toBe(true);
        }
      }
    }
  }
}

/**
 * Compile a source diagram into an NNTree JSON file inside outDir.
 * Returns the path to the written JSON file.
 */
export function compileDiagramToFile(
  name: string,
  outDir: string,
  options?: { diagramsDir?: string },
): string {
  const diagramPath = resolve(options?.diagramsDir ?? DIAGRAMS_DIR, `${name}.json`);
  const content = readFileSync(diagramPath, "utf-8");

  const diagram = new Diagram();
  diagram.importFromJson(content);

  const jsonStr = new NNTree(diagram).toJson();
  const jsonPath = join(outDir, `${name}.json`);
  writeFileSync(jsonPath, jsonStr, "utf-8");
  return jsonPath;
}

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

/**
 * Create a caller-owned temporary directory. The caller is responsible for
 * registering it (e.g. in the test's afterEach via conditionalCleanup) and for
 * passing it as workDir to compileDiagramToFile/runConvert/runTraining so all
 * helper-generated artifacts live under it. Auto-cleaned unless
 * NNM_KEEP_TEMP=true.
 */
export function tempDir(): string {
  return mkdtempSync(join(tmpdir(), "nnm-integration-"));
}

/** Recursively remove a temp directory (safely). */
export function cleanupTempDir(dir: string): void {
  if (!dir || !dir.startsWith(tmpdir())) return;
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }
}

/**
 * Conditionally clean up a temp dir — only if NNM_KEEP_TEMP is not set.
 * Useful to call in afterEach / afterAll.
 */
export function conditionalCleanup(dir: string): void {
  if (!process.env.NNM_KEEP_TEMP) {
    cleanupTempDir(dir);
  }
}

// ---------------------------------------------------------------------------
// Python subprocess management
// ---------------------------------------------------------------------------

/**
 * Run a command via `uv run` (Python in uv-managed venv).
 * Throws on timeout. Returns { exitCode, stdout, stderr }.
 */
export function uvRun(
  args: string[],
  opts?: {
    cwd?: string;
    env?: Record<string, string>;
    timeout?: number;
  },
): SpawnResult {
  const cwd = opts?.cwd || CONVERTED_DIR;
  const timeout = opts?.timeout || 120_000;

  const result = spawnSync("uv", ["run", ...args], {
    cwd,
    env: { ...process.env, ...opts?.env },
    timeout,
    encoding: "utf-8",
    maxBuffer: 10 * 1024 * 1024, // 10MB
  });

  return {
    exitCode: result.status ?? -1,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    signal: result.signal,
    error: result.error?.message,
  };
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/**
 * Run convert.py on a JSON file (NNTree or Svelte Flow JSON).
 * Returns the path to the generated Hydra config directory.
 *
 * Ownership model: helpers never create un-owned temp parents. By default the
 * config directory is placed next to the input JSON file, so the caller's
 * teardown of its own workDir (created with tempDir()) also removes the
 * generated Hydra configs. An explicit `outputDir` transfers ownership to the
 * caller.
 */
export function runConvert(
  jsonPath: string,
  options?: {
    outputDir?: string;
    numClasses?: number;
    dataset?: string;
    timeout?: number;
  },
): string {
  const outDir =
    options?.outputDir || resolve(dirname(jsonPath), "hydra_config");
  if (!options?.outputDir) {
    // Guard: never silently write into a non-temporary parent (e.g. a repo
    // fixture) — the caller must own the destination explicitly.
    const parent = dirname(outDir);
    if (parent !== tmpdir() && !parent.startsWith(tmpdir() + sep)) {
      throw new Error(
        "runConvert: refusing to write Hydra configs into a non-temporary " +
          `parent (${parent}). Pass an explicit caller-owned outputDir.`,
      );
    }
  }
  mkdirSync(outDir, { recursive: true });

  const args = ["python", "src/convert.py", jsonPath, outDir];
  if (options?.numClasses !== undefined) {
    args.push("--num-classes", String(options.numClasses));
  }
  if (options?.dataset) {
    args.push("--dataset", options.dataset);
  }

  const result = uvRun(args, { timeout: options?.timeout ?? 120_000 });
  if (result.exitCode !== 0) {
    throw new Error(
      `convert.py failed (exit ${result.exitCode}):\n` +
        `STDERR:\n${result.stderr}\nSTDOUT:\n${result.stdout}`,
    );
  }

  return outDir;
}

/** Expected YAML files generated by convert.py */
export const EXPECTED_YAML_FILES = [
  "base.yaml",
  "net/custom_sequence.yaml",
  "optimizer/adam.yaml",
  "trainer/default.yaml",
  "wandb/wandb.yaml",
  "dataset/dataset.yaml",
  "early_stopping/default.yaml",
];

/**
 * The generated trainer runs with `hydra.job.chdir=true`, so the dataset root
 * ("data") resolves inside the temporary output directory. Without a local
 * copy, every training run re-downloads MNIST over the network into that temp
 * dir — slow and flaky. When the repo-local cache (converted/data, gitignored)
 * exists, expose it under the output dir so training is offline and
 * deterministic. No-op when the cache is absent (behavior unchanged).
 */
export function seedDatasetCache(outputDir: string): void {
  const localData = resolve(CONVERTED_DIR, "data");
  const targetData = join(outputDir, "data");
  if (existsSync(resolve(localData, "MNIST")) && !existsSync(targetData)) {
    try {
      symlinkSync(localData, targetData, "dir");
    } catch {
      // Fall back to a copy when symlinks are unavailable.
      cpSync(localData, targetData, { recursive: true });
    }
  }
}

/**
 * Run main.py training with a Hydra config dir.
 * Every run writes checkpoints and weights into its own output directory.
 *
 * The default output dir is `dirname(cfgDir)/training-output`. Under the
 * ownership model, cfgDir is produced by runConvert inside the caller-owned
 * workDir, so training outputs/weights/checkpoints stay inside the caller's
 * teardown scope.
 */
export function runTraining(
  cfgDir: string,
  options?: {
    maxEpochs?: number;
    fastDevRun?: boolean;
    device?: string;
    dataset?: string;
    outputDir?: string;
    timeout?: number;
  },
): TrainResult {
  const device = options?.device || process.env.NNM_DEVICE || "cpu";
  const outputDir = options?.outputDir || resolve(dirname(cfgDir), "training-output");
  mkdirSync(outputDir, { recursive: true });
  seedDatasetCache(outputDir);
  const overrides: string[] = [
    `trainer.accelerator=${device}`,
    `+trainer.devices=${device === "gpu" ? "auto" : "1"}`,
    `+trainer.enable_progress_bar=false`,
    `+trainer.default_root_dir=${outputDir}`,
    `+wandb.mode=${process.env.NNM_WANDB_MODE || "disabled"}`,
    `hydra.run.dir=${outputDir}`,
    "hydra.job.chdir=true",
  ];

  if (options?.fastDevRun) {
    overrides.push("+trainer.fast_dev_run=true");
  } else if (options?.maxEpochs) {
    overrides.push(`trainer.max_epochs=${options.maxEpochs}`);
  }

  if (options?.dataset) {
    overrides.push(`dataset=${options.dataset}`);
  }

  const args = [
    "python",
    "src/main.py",
    "--config-path",
    cfgDir,
    "--config-name",
    "base",
    ...overrides,
  ];

  const result = uvRun(args, { timeout: options?.timeout ?? 600_000 });

  return {
    ...result,
    outputDir,
    weightsPath: join(outputDir, "weights.pt"),
  };
}

/** Compose the generated Hydra config without starting a training run. */
export function composeHydraConfig(
  cfgDir: string,
  overrides: string[] = [],
): SpawnResult {
  return uvRun([
    "python",
    "src/main.py",
    "--config-path",
    cfgDir,
    "--config-name",
    "base",
    "--cfg",
    "job",
    ...overrides,
  ]);
}

/**
 * Run infer.py with a trained checkpoint.
 * Returns SpawnResult.
 *
 * The default prediction output lands next to the caller-owned cfgDir (which
 * is the caller's workDir when runConvert used its default), so normal
 * teardown removes it too.
 */
export function runInference(
  cfgDir: string,
  ckptPath: string,
  options?: {
    outputPath?: string;
    imageDir?: string;
    configName?: string;
    timeout?: number;
  },
): SpawnResult {
  const outPath =
    options?.outputPath || join(dirname(cfgDir), "predictions.json");

  const args = [
    "python",
    "src/infer.py",
    "--config-path",
    cfgDir,
    "--config-name",
    options?.configName || "base",
    "--weights",
    ckptPath,
    "--output",
    outPath,
  ];

  if (options?.imageDir) {
    args.push("--image-dir", options.imageDir);
  }

  return uvRun(args, { timeout: options?.timeout ?? 300_000 });
}

/**
 * Check whether the uv-managed Python environment is available.
 * Returns true if `uv run python --version` succeeds.
 */
export function isPythonAvailable(): boolean {
  try {
    const result = uvRun(["python", "--version"], { timeout: 10_000 });
    return result.exitCode === 0;
  } catch {
    return false;
  }
}

/**
 * Resolve whether the Python prerequisites are available for a Python-backed
 * integration tier.
 *
 * - When `NNM_REQUIRE_PYTHON=true` (CI-required mode) a missing runtime is an
 *   error: this throws instead of silently skipping.
 * - Otherwise (local lenient mode) it returns false so the tier can
 *   `describe.skip` with an explicit reason.
 */
export function requirePython(): boolean {
  const available = isPythonAvailable();
  if (!available && process.env.NNM_REQUIRE_PYTHON === "true") {
    throw new Error(
      "Python prerequisites are unavailable (`uv run python --version` failed) while " +
        "NNM_REQUIRE_PYTHON=true. This tier is required: install uv and run `uv sync` " +
        "inside converted/ instead of silently skipping.",
    );
  }
  return available;
}

/**
 * Find a .ckpt file in a checkpoint directory (searches recursively).
 * Returns the first .ckpt found, or throws.
 */
export function findCheckpoint(dir: string): string {
  const files = readdirSync(dir, { recursive: true });
  const ckpt = files.find((f) => f.toString().endsWith(".ckpt"));
  if (!ckpt) {
    throw new Error(`No checkpoint file found in ${dir}`);
  }
  return resolve(dir, ckpt.toString());
}
