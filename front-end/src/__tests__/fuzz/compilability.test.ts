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

import fc from "fast-check";
import { afterAll } from "vitest";
import { Diagram } from "../../Diagram.svelte";
import { NNTree } from "../../conversion/nnTree";
import { stubWindow, unstubWindow, node, edge } from "../helpers";
import { MODULE_STEREOS, LOSS_STEREOS, assertTreeInvariants } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

// ── fast-check arbitraries ────────────────────────────────────────────

/** Generate a random valid graph: 1 Input + N intermediates + 1 Loss. */
const validGraphArb = fc.record({
  modules: fc.array(fc.constantFrom(...MODULE_STEREOS), { minLength: 0, maxLength: 5 }),
  lossType: fc.constantFrom(...LOSS_STEREOS),
  branch: fc.boolean(),
});

/** Generate graph specs with varying Input counts (0, 1, or 2+). */
const inputCardinalityArb = fc.record({
  modules: fc.array(fc.constantFrom(...MODULE_STEREOS), { minLength: 1, maxLength: 4 }),
  lossType: fc.constantFrom(...LOSS_STEREOS),
  // 0 = no Input, 1 = one Input (valid), 2 = multiple Inputs
  inputCount: fc.integer({ min: 0, max: 2 }),
});

// ── Build a Diagram from a spec ──────────────────────────────────────

function buildValidGraph(spec: { modules: string[]; lossType: string; branch: boolean }): Diagram {
  const d = new Diagram();

  const inputNode = node("input", "Input", "Input_0", {}, { isInput: true });
  const nodes = [inputNode];
  const edges: Array<{ id: string; source: string; target: string }> = [];

  let prevId = "input";
  for (let i = 0; i < spec.modules.length; i++) {
    const id = `m${i}`;
    const stereo = spec.modules[i];
    nodes.push(node(id, stereo, `${stereo}_${i}`));
    edges.push({ id: `e_${prevId}_${id}`, source: prevId, target: id });
    prevId = id;
  }

  if (spec.branch && spec.modules.length >= 2) {
    const branchNodeId = `branch_0`;
    nodes.push(node(branchNodeId, "Addition", "Addition_0", {}, { type: "join" }));
    edges.push({ id: `e_branch_from_0`, source: `m0`, target: branchNodeId, ...{ targetHandle: "in-0" } });
    edges.push({ id: `e_branch_from_1`, source: `m1`, target: branchNodeId, ...{ targetHandle: "in-1" } });
    edges.push({ id: `e_branch_to_loss`, source: branchNodeId, target: `loss`, ...{ sourceHandle: "out" } });
  }

  // Loss node
  const prev = prevId;
  if (!spec.branch || spec.modules.length < 2) {
    edges.push({ id: `e_${prev}_loss`, source: prev, target: "loss" });
  }
  nodes.push(node("loss", spec.lossType, `${spec.lossType}_0`, {}, { isLoss: true }));

  d.nodes = nodes as any;
  d.edges = edges as any;
  return d;
}

function buildWithInputCount(spec: {
  modules: string[]; inputCount: number; lossType: string;
}): Diagram {
  const d = new Diagram();
  const nodes: any[] = [];
  const edges: any[] = [];

  for (let i = 0; i < spec.inputCount; i++) {
    nodes.push(node(`input${i}`, "Input", `Input_${i}`, {}, { isInput: true }));
  }
  if (spec.inputCount === 0) {
    nodes.push(node("first", spec.modules[0], "First", {}));
  }

  let prevId = spec.inputCount > 0 ? "input0" : "first";
  for (let i = 0; i < spec.modules.length; i++) {
    const id = `m${i}`;
    nodes.push(node(id, spec.modules[i], `${spec.modules[i]}_${i}`));
    edges.push({ id: `e_${prevId}_${id}`, source: prevId, target: id });
    prevId = id;
  }

  const lossId = "loss";
  nodes.push(node(lossId, spec.lossType, `${spec.lossType}_0`, {}, { isLoss: true }));
  edges.push({ id: `e_${prevId}_${lossId}`, source: prevId, target: lossId });

  d.nodes = nodes as any;
  d.edges = edges as any;
  return d;
}

// ── Fuzzer #1 — Graph Compilability ─────────────────────────────────

describe("Fuzzer #1 — Graph Compilability", () => {

  it("Every valid graph compiles into a correct NNTree", () => {
    fc.assert(
      fc.property(validGraphArb, (spec) => {
        const d = buildValidGraph(spec);
        const tree = new NNTree(d);
        assertTreeInvariants(tree, d);
      }),
      { numRuns: 500 },
    );
  });

  it("Graphs without exactly one Input throw controlled error", () => {
    fc.assert(
      fc.property(inputCardinalityArb, (spec) => {
        const d = buildWithInputCount(spec);
        // NNTree only enforces: exactly 1 Input node.
        // Loss is absorbed as the last childless node — no cardinality check.
        if (spec.inputCount !== 1) {
          expect(() => new NNTree(d)).toThrow(Error);
        } else {
          expect(() => new NNTree(d)).not.toThrow();
        }
      }),
      { numRuns: 500 },
    );
  });
});
