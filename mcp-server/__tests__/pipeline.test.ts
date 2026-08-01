import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { buildTrainingArgs, executeConversion } from "../src/pipeline";

describe("buildTrainingArgs", () => {
  it("uses Hydra overrides and isolates all training artifacts", () => {
    const args = buildTrainingArgs(
      {
        configDir: "/tmp/config",
        configName: "base",
        device: "cpu",
        maxEpochs: 1,
      },
      "/tmp/run",
    );

    expect(args).toEqual([
      "src/main.py",
      "--config-path",
      "/tmp/config",
      "--config-name",
      "base",
      "trainer.max_epochs=1",
      "trainer.accelerator=cpu",
      "+trainer.devices=1",
      "hydra.run.dir=/tmp/run",
      "hydra.job.chdir=true",
      "+trainer.default_root_dir=/tmp/run",
      "+trainer.enable_progress_bar=false",
      "+wandb.mode=disabled",
    ]);
    expect(args).not.toContain("--max-epochs");
    expect(args).not.toContain("--device");
  });

  it("converts from the repository root, as the documented tsx development command does", async () => {
    const originalCwd = process.cwd();
    const repositoryRoot = resolve(import.meta.dirname, "../..");
    const outputDir = mkdtempSync(join(tmpdir(), "nnm-pipeline-cwd-"));
    const nntree = {
      root: "input",
      lossNode: {
        type: "module",
        name: "loss",
        stereotype: "CrossEntropyLoss",
        pythonClassName: "nn.CrossEntropyLoss",
        taskType: "classification",
        params: {},
      },
      nodes: {
        input: {
          id: "input",
          children: [],
          data: {
            type: "module",
            name: "input",
            stereotype: "Input",
            pythonClassName: "None",
            params: {},
          },
        },
      },
    };

    try {
      process.chdir(repositoryRoot);
      await expect(executeConversion(JSON.stringify(nntree), { outputDir })).resolves.toMatchObject({
        success: true,
        taskType: "classification",
      });
    } finally {
      process.chdir(originalCwd);
      rmSync(outputDir, { recursive: true, force: true });
    }
  });
});
