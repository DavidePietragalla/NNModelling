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
 *
 * Differentiability invariant (plan.md §T2, invariant 4):
 * a canonical generated model must support a real backward pass —
 * finite loss, finite/nonzero gradients and at least one parameter update
 * after an optimizer step. Task-appropriate targets are chosen per model:
 * classification -> random class indices, autoencoder -> the input itself.
 * No loss-decrease assertion is made (it would be flaky in one step).
 */

import { describe, it, expect, afterAll, afterEach } from "vitest";
import { writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { stubWindow, unstubWindow } from "../helpers";
import {
  loadManifest,
  getValidationModels,
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
 * Python script that runs a real forward/backward/optimizer-step cycle on the
 * generated model and reports numeric invariants as JSON.
 */
function generateBackwardScript(convertedSrcPath: string): string {
  return `
import sys
import json
from hydra import compose, initialize_config_dir
import torch

sys.path.insert(0, ${JSON.stringify(convertedSrcPath)})
from net.base import Net

cfg_dir = sys.argv[1]
task_type = sys.argv[2]
num_classes = int(sys.argv[3])
batch = int(sys.argv[4])

with initialize_config_dir(config_dir=cfg_dir):
    cfg = compose(config_name="base")

net = Net(cfg)
net.train()

torch.manual_seed(0)
x = torch.randn(batch, 1, 28, 28)

y_hat = net(x)
if task_type == "classification":
    y = torch.randint(0, num_classes, (y_hat.shape[0],))
else:
    y = x

loss = net.loss_fn(y_hat, y)
loss_value = float(loss.detach())
loss.backward()

grads = [p.grad for p in net.parameters() if p.grad is not None]
grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
grad_nonzero = any(bool((g != 0).any()) for g in grads)

before = {k: v.clone() for k, v in net.state_dict().items()}
optimizer = net.configure_optimizers()
optimizer.zero_grad()
loss = net.loss_fn(net(x), y)
loss.backward()
optimizer.step()
after = net.state_dict()
changed = [k for k in before if not torch.equal(before[k], after[k])]

result = {
    "loss_finite": bool(torch.isfinite(loss).item()),
    "loss_value": loss_value,
    "num_grads": len(grads),
    "grads_finite": grad_finite,
    "grads_nonzero": grad_nonzero,
    "params_changed": len(changed),
    "params_total": len(before),
    "success": True,
}
print(json.dumps(result))
`;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  // Backward coverage targets the validation models (mninst and
  // autoencoder_mnist) — the models selected for this milestone.
  const targets = getValidationModels(manifest);

  if (!PYTHON_AVAILABLE) {
    describe.skip("Backward — Python environment unavailable", () => {
      it("requires uv and the converted Python environment", () => {});
    });
  } else {
    describe.each(targets)(
      "Backward: $name",
      ({ name, entry }: NamedEntry) => {
        const tmpDirs: string[] = [];

        afterEach(() => {
          for (const d of tmpDirs) {
            conditionalCleanup(d);
          }
        });

        it(
          "finite loss, finite/nonzero gradients, parameter update",
          { timeout: 300_000 },
          () => {
            const workDir = tempDir();
            tmpDirs.push(workDir);

            const jsonPath = compileDiagramToFile(name, workDir);
            const cfgDir = runConvert(jsonPath, {
              numClasses: entry.numClasses,
              dataset: entry.dataset,
            });

            const scriptPath = join(workDir, "backward.py");
            writeFileSync(
              scriptPath,
              generateBackwardScript(resolve(CONVERTED_DIR, "src")),
              "utf-8",
            );

            const batch = 4;
            const result = uvRun(
              [
                "python",
                scriptPath,
                cfgDir,
                entry.taskType,
                String(entry.numClasses ?? 0),
                String(batch),
              ],
              { timeout: 180_000 },
            );

            if (result.exitCode !== 0) {
              throw new Error(
                `Backward script failed for ${name} (exit ${result.exitCode}):\n` +
                  `STDERR:\n${result.stderr}\nSTDOUT:\n${result.stdout}`,
              );
            }

            const output = JSON.parse(result.stdout.trim());
            expect(output.success).toBe(true);
            expect(output.loss_finite).toBe(true);
            expect(output.num_grads).toBeGreaterThan(0);
            expect(output.grads_finite).toBe(true);
            expect(output.grads_nonzero).toBe(true);
            expect(output.params_changed).toBeGreaterThan(0);
            expect(output.params_changed).toBeLessThanOrEqual(output.params_total);
          },
        );
      },
    );
  }
} else {
  describe.skip("Backward tier disabled (runs under forward tier)", () => {
    it("runs only when NNM_TIER is forward or all", () => {});
  });
}
