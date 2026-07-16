import { describe, expect, it } from "vitest";
import { buildTrainingArgs } from "../src/pipeline";

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
});
