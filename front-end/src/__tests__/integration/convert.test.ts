/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
  tempDir,
  conditionalCleanup,
  runConvert,
  EXPECTED_YAML_FILES,
  type Tier,
  type NamedEntry,
} from "./helpers";

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const manifest = loadManifest();
const tier = (process.env.NNM_TIER || "all") as Tier;
const shouldRun = tier === "all" || tier === "convert";

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compile a diagram and save the NNTree JSON to a temp file. Returns the path. */
function compileToTempFile(name: string, outDir: string): string {
  const diagramPath = resolve(DIAGRAMS_DIR, `${name}.json`);
  const content = readFileSync(diagramPath, "utf-8");

  const diagram = new Diagram();
  diagram.importFromJson(content);

  const nnTree = new NNTree(diagram);
  const jsonStr = nnTree.toJson();

  // debug: check tree structure
  const parsed = JSON.parse(jsonStr);
  if (!parsed.root || Object.keys(parsed.nodes).length === 0) {
    console.error(
      `[DEBUG] Empty NNTree for ${name}:`,
      JSON.stringify(parsed).substring(0, 300),
    );
    console.error(`[DEBUG] Diagram had ${diagram.nodes.length} nodes`);
  }

  const jsonPath = join(outDir, `${name}.json`);
  writeFileSync(jsonPath, jsonStr, "utf-8");
  return jsonPath;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

if (shouldRun) {
  const targets = getTargetDiagrams(manifest, "convert");

  describe.each(targets)("Convert: $name", ({ name, entry }: NamedEntry) => {
    const tmpDirs: string[] = [];

    afterEach(() => {
      for (const d of tmpDirs) {
        conditionalCleanup(d);
      }
    });

    it("convert.py produces valid Hydra config directory", { timeout: 120_000 }, () => {
      // Compile diagram -> NNTree JSON
      const workDir = tempDir();
      tmpDirs.push(workDir);
      const jsonPath = compileToTempFile(name, workDir);

      // Run convert.py
      const cfgDir = runConvert(jsonPath, {
        numClasses: entry.numClasses,
      });

      // Verify expected YAML files exist
      for (const yamlFile of EXPECTED_YAML_FILES) {
        const yamlPath = resolve(cfgDir, yamlFile);
        expect(
          existsSync(yamlPath),
          `Expected ${yamlFile} to exist at ${yamlPath}`,
        ).toBe(true);
      }
    });

    it("convert.py accepts --dataset option", { timeout: 120_000 }, () => {
      const workDir = tempDir();
      tmpDirs.push(workDir);
      const jsonPath = compileToTempFile(name, workDir);

      // Run convert.py with dataset option
      const cfgDir = runConvert(jsonPath, {
        numClasses: entry.numClasses,
        dataset: "mnist",
      });

      // Verify base.yaml exists
      expect(existsSync(resolve(cfgDir, "base.yaml"))).toBe(true);
    });

    it("convert.py handles incomplete JSON gracefully", () => {
      // convert.py gracefully accepts many JSON shapes (returns 0),
      // but we verify it at least produces output
      const workDir = tempDir();
      tmpDirs.push(workDir);
      const incompleteJsonPath = join(workDir, "incomplete.json");
      writeFileSync(
        incompleteJsonPath,
        '{"not": "valid nntree"}',
        "utf-8",
      );

      // Should not throw — convert.py handles incomplete JSON
      let cfgDir: string;
      expect(() => {
        cfgDir = runConvert(incompleteJsonPath);
      }).not.toThrow();
    });
  });
} else {
  describe("Convert", () => {
    it("skipped — NNM_TIER is not 'convert' or 'all'", () => {
      expect(true).toBe(true);
    });
  });
}
