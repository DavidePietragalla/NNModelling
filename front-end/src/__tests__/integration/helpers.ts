import { spawnSync } from "node:child_process";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  mkdtempSync,
  rmSync,
  existsSync,
  readFileSync,
  mkdirSync,
  readdirSync,
} from "node:fs";
import { tmpdir } from "node:os";

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

/** Manifest path */
export const MANIFEST_PATH = resolve(EXAMPLES_DIR, "manifest.json");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DiagramEntry {
  format: "svelte-flow" | "nntree";
  inputShape: number[];
  inputType?: "float" | "int";
  numClasses?: number;
  taskType: "classification" | "regression";
  trainable: boolean;
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
}

export interface NamedEntry {
  name: string;
  entry: DiagramEntry;
}

export interface TrainResult {
  exitCode: number;
  ckptDir: string;
  stdout: string;
  stderr: string;
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
      .filter(([_, entry]) => entry.trainable)
      .map(([name, entry]) => ({ name, entry }));
  }

  return Object.entries(diagrams).map(([name, entry]) => ({ name, entry }));
}

// ---------------------------------------------------------------------------
// Temp directory management
// ---------------------------------------------------------------------------

/** Create a temporary directory. Auto-cleaned unless NNM_KEEP_TEMP=true. */
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
  };
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/**
 * Run convert.py on a JSON file (NNTree or Svelte Flow JSON).
 * Returns the path to the generated Hydra config directory.
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
    options?.outputDir || resolve(tempDir(), "hydra_config");
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
 * Run main.py training with a Hydra config dir.
 * Returns { exitCode, ckptDir, stdout, stderr }.
 */
export function runTraining(
  cfgDir: string,
  options?: {
    maxEpochs?: number;
    fastDevRun?: boolean;
    device?: string;
    dataset?: string;
    timeout?: number;
  },
): TrainResult {
  const device = options?.device || process.env.NNM_DEVICE || "cpu";
  const overrides: string[] = [
    `trainer.accelerator=${device}`,
    `trainer.devices=${device === "gpu" ? "auto" : "1"}`,
    `+trainer.enable_progress_bar=false`,
    `+wandb.mode=${process.env.NNM_WANDB_MODE || "disabled"}`,
  ];

  if (options?.fastDevRun) {
    overrides.push("trainer.fast_dev_run=true");
  } else if (options?.maxEpochs) {
    overrides.push(`trainer.max_epochs=${options.maxEpochs}`);
  }

  if (options?.dataset) {
    overrides.push(`dataset=${options.dataset}`);
  }

  const args = [
    "python",
    "src/main.py",
    `--config-dir=${cfgDir}`,
    "--config-name=base",
    ...overrides,
  ];

  const result = uvRun(args, { timeout: options?.timeout ?? 600_000 });

  // Training saves weights.pt to CWD (the directory uv is run from)
  const ckptDir = CONVERTED_DIR;
  return {
    exitCode: result.exitCode,
    ckptDir,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

/**
 * Run infer.py with a trained checkpoint.
 * Returns SpawnResult.
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
    options?.outputPath || join(tmpdir(), "nnm-predictions.json");

  const args = [
    "python",
    "src/infer.py",
    `--config-path=${cfgDir}`,
    `--config-name=${options?.configName || "base"}`,
    `--weights=${ckptPath}`,
    `--output=${outPath}`,
  ];

  if (options?.imageDir) {
    args.push(`--image-dir=${options.imageDir}`);
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
