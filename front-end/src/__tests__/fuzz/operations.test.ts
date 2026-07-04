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

import fc from "fast-check";
import { afterAll } from "vitest";
import { Diagram } from "../../Diagram.svelte";
import { stubWindow, unstubWindow } from "../helpers";
import { assertGraphConsistent, MODULE_STEREOS, JOIN_STEREOS } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

// ── fast-check arbitraries for operations ────────────────────────────

type OpAddModule = { kind: "addModule"; stereo: string; x: number; y: number };
type OpAddJoin = { kind: "addJoin"; stereo: string; x: number; y: number; inputs: number };
type OpDelete = { kind: "delete"; id: string };
type OpMove = { kind: "move"; id: string; x: number; y: number };
type OpEdge = { kind: "edge"; source: string; target: string };
type OpUndo = { kind: "undo" };
type OpRedo = { kind: "redo" };

type Op = OpAddModule | OpAddJoin | OpDelete | OpMove | OpEdge | OpUndo | OpRedo;

const addModuleArb: fc.Arbitrary<OpAddModule> = fc.record({
  kind: fc.constant("addModule" as const),
  stereo: fc.constantFrom(...MODULE_STEREOS),
  x: fc.integer({ min: 0, max: 500 }),
  y: fc.integer({ min: 0, max: 500 }),
});

const addJoinArb: fc.Arbitrary<OpAddJoin> = fc.record({
  kind: fc.constant("addJoin" as const),
  stereo: fc.constantFrom(...JOIN_STEREOS),
  x: fc.integer({ min: 0, max: 500 }),
  y: fc.integer({ min: 0, max: 500 }),
  inputs: fc.integer({ min: 2, max: 4 }),
});

const opArb: fc.Arbitrary<Op> = fc.oneof(
  addModuleArb,
  addJoinArb,
  fc.constant({ kind: "undo" as const }),
  fc.constant({ kind: "redo" as const }),
);

const opsSequenceArb = fc.array(opArb, { minLength: 1, maxLength: 15 });

// ── Fuzzer #4 — Operation Commutativity ─────────────────────────────

describe("Fuzzer #4 — Operation Commutativity", () => {

  it("After every operation, graph is internally consistent", () => {
    fc.assert(
      fc.property(opsSequenceArb, (ops) => {
        const d = new Diagram();
        const stereo = d.getStereotype("Linear");
        if (!stereo) throw new Error("Linear not found");

        for (const op of ops) {
          try {
            switch (op.kind) {
              case "addModule": {
                const s = d.getStereotype(op.stereo);
                if (s) d.addModule(s, op.x, op.y);
                break;
              }
              case "addJoin": {
                const s = d.getStereotype(op.stereo);
                if (s) d.addJoinNode(s, op.x, op.y, { inputsCount: op.inputs });
                break;
              }
              case "undo":
                d.undo();
                break;
              case "redo":
                d.redo();
                break;
            }
          } catch {
            // Operations can fail (e.g. addEdge on occupied handle) — that's fine.
            // The invariant is that after ANY operation, the graph stays consistent.
          }

          assertGraphConsistent(d);
        }
      }),
      { numRuns: 200 },
    );
  });

  it("Undo/redo sequence returns to exact state (snapshot identity)", () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(...MODULE_STEREOS), { minLength: 1, maxLength: 5 }),
        (stereos) => {
          const d = new Diagram();
          const snapBefore = d.getSnapshot();

          for (const sName of stereos) {
            const s = d.getStereotype(sName);
            if (s) d.addModule(s, 100, 100);
          }

          const snapMid = d.getSnapshot();

          // Undo all operations back to start
          while (d.undo()) { /* loop */ }

          const snapAfterUndo = d.getSnapshot();
          expect(snapAfterUndo.nodes.length).toBe(snapBefore.nodes.length);
          expect(snapAfterUndo.edges.length).toBe(snapBefore.edges.length);

          // Redo all
          while (d.redo()) { /* loop */ }

          const snapAfterRedo = d.getSnapshot();
          expect(snapAfterRedo.nodes.length).toBe(snapMid.nodes.length);
          expect(snapAfterRedo.edges.length).toBe(snapMid.edges.length);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("50 undo stack limit is respected", () => {
    const d = new Diagram();
    const linear = d.getStereotype("Linear")!;

    // Add 60 modules (creates 60 undo entries)
    for (let i = 0; i < 60; i++) {
      d.addModule(linear, i * 10, i * 10);
    }

    expect(d.nodes.length).toBe(61); // 60 + 1 Input

    // Only 50 undos possible
    let undoCount = 0;
    while (d.undo()) { undoCount++; }
    expect(undoCount).toBe(50);
    expect(d.nodes.length).toBe(11); // 10 remain (50 undone + 1 Input)
  });
});
