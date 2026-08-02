import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
    const originalPath = process.env.PATH;
    const markerEnvName = "NNM_PIPELINE_TEST_CWD_MARKER";
    const originalMarker = process.env[markerEnvName];
    const repositoryRoot = resolve(import.meta.dirname, "../..");
    const toolDir = mkdtempSync(join(tmpdir(), "nnm-pipeline-uv-"));
    const outputDir = mkdtempSync(join(tmpdir(), "nnm-pipeline-cwd-"));
    const markerPath = join(toolDir, "cwd-marker");
    const fakeUvPath = join(toolDir, "uv");
    writeFileSync(
      fakeUvPath,
      `#!/bin/sh
printf '%s' "$PWD" > "$${markerEnvName}"
exit 0
`,
      "utf-8",
    );
    chmodSync(fakeUvPath, 0o755);
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
      process.env.PATH = `${toolDir}:${originalPath ?? ""}`;
      process.env[markerEnvName] = markerPath;
      process.chdir(repositoryRoot);
      await expect(executeConversion(JSON.stringify(nntree), { outputDir })).resolves.toMatchObject({
        success: true,
        taskType: "classification",
      });
      expect(readFileSync(markerPath, "utf-8")).toBe(resolve(repositoryRoot, "converted"));
    } finally {
      process.chdir(originalCwd);
      if (originalPath === undefined) delete process.env.PATH;
      else process.env.PATH = originalPath;
      if (originalMarker === undefined) delete process.env[markerEnvName];
      else process.env[markerEnvName] = originalMarker;
      rmSync(toolDir, { recursive: true, force: true });
      rmSync(outputDir, { recursive: true, force: true });
    }
  });
});
