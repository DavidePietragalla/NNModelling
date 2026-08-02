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
 * Regression tests for the temp-directory ownership model (review finding
 * H1). Helpers must never create un-owned nnm-integration-* temp parents:
 * every generated artifact lives under the caller-owned workDir and is
 * removed by the caller's normal teardown. NNM_KEEP_TEMP stays the explicit,
 * intentional opt-out from cleanup.
 */

import { describe, it, expect, afterAll } from "vitest";
import { existsSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { stubWindow, unstubWindow } from "../helpers";
import {
  PROJECT_ROOT,
  tempDir,
  cleanupTempDir,
  conditionalCleanup,
  runConvert,
  compileDiagramToFile,
  type Tier,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const tier = (process.env.NNM_TIER || "all") as Tier;

// Needed by compileDiagramToFile (Diagram construction).
stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Keep-temp semantics (pure TypeScript — smoke tier)
// ---------------------------------------------------------------------------

if (tier === "all" || tier === "smoke") {
  describe("temp ownership: keep-temp semantics", () => {
    it("conditionalCleanup removes a registered dir by default", () => {
      const dir = tempDir();
      expect(existsSync(dir)).toBe(true);

      conditionalCleanup(dir);

      expect(existsSync(dir)).toBe(false);
    });

    it("NNM_KEEP_TEMP=true keeps the dir intentionally", () => {
      const dir = tempDir();
      process.env.NNM_KEEP_TEMP = "true";
      try {
        conditionalCleanup(dir);
        // Intentional opt-out: teardown must leave the dir for inspection.
        expect(existsSync(dir)).toBe(true);
      } finally {
        delete process.env.NNM_KEEP_TEMP;
        cleanupTempDir(dir);
      }
    });
  });
} else {
  describe.skip("Temp ownership (keep-temp) disabled", () => {
    it("runs only when NNM_TIER is smoke or all", () => {});
  });
}

// ---------------------------------------------------------------------------
// runConvert ownership (real convert.py — convert tier)
// ---------------------------------------------------------------------------

if (tier === "all" || tier === "convert") {
  describe("temp ownership: runConvert", () => {
    it("places configs under the caller-owned workDir and normal teardown reaps them", () => {
      const workDir = tempDir();
      let cfgDir: string;
      try {
        const jsonPath = compileDiagramToFile("mninst", workDir);
        cfgDir = runConvert(jsonPath, {
          numClasses: 10,
          dataset: "dataset.mnist.MNISTDataset",
        });

        // Ownership: the config parent is the caller-owned workDir — the
        // helper introduced no independent temp parent.
        expect(dirname(cfgDir)).toBe(workDir);
        expect(cfgDir.startsWith(workDir)).toBe(true);
        expect(existsSync(join(cfgDir, "base.yaml"))).toBe(true);
      } finally {
        cleanupTempDir(workDir);
      }

      // Normal teardown of the owned workDir removes the generated configs
      // and everything else the chain produced inside it.
      expect(existsSync(join(workDir, "hydra_config"))).toBe(false);
      expect(existsSync(workDir)).toBe(false);
    });

    it("refuses to write configs into a non-temporary parent without outputDir", () => {
      // A repo fixture is not caller-owned: without an explicit outputDir the
      // helper must throw instead of silently writing into the repository.
      const repoNnTree = resolve(PROJECT_ROOT, "examples", "nntrees", "mninst_skip.json");
      expect(() => runConvert(repoNnTree, { numClasses: 10 })).toThrow(
        /non-temporary|outputDir/i,
      );
    });
  });
} else {
  describe.skip("Temp ownership (runConvert) disabled", () => {
    it("runs only when NNM_TIER is convert or all", () => {});
  });
}
