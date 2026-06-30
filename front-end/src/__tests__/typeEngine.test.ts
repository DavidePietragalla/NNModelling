/**
 * @file Type inference engine unit tests.
 *
 * NOTE: Phase 1 only supports wildcards at the last position in a pattern.
 *       Linear's pattern [B, *, in_features] has wildcard in the middle,
 *       so tests depending on Linear type inference are skipped until
 *       frontend-3 enables mid-pattern wildcard support.
 */

import { describe, it, expect, afterAll } from "vitest";
import { Diagram } from "../Diagram.svelte";
import { TypeEngine } from "../conversion/typeEngine";
import type { TypeResult } from "../conversion/tensortypes";
import {
  stubWindow,
  unstubWindow,
  node,
  edge,
  expectTypeSuccess,
  expectOutputShape,
  expectTypeError,
} from "./helpers";

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Group 1 — Happy Path: Simple Sequential Chains
// ---------------------------------------------------------------------------

describe("TypeEngine — Happy Path", () => {
  it("1.2: Input → ReLU preserves shape through shape-preserving module", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    const reluStereo = d.stereotypes.find((s) => s.name === "ReLU")!;
    d.addModule(reluStereo, 200, 0);
    const reluId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, reluId));

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);

    // Input annotation
    expect(result.annotations.has(inputId)).toBe(true);
    const inputAnn = result.annotations.get(inputId)!;
    expect(inputAnn.inputType).toBeUndefined();
    expect(inputAnn.outputType.shape[0]).toEqual({
      kind: "symbolic",
      name: "B",
    });
    expect(inputAnn.outputType.shape[1]).toEqual({
      kind: "const",
      value: 784,
    });
    expect(inputAnn.outputType.dtype).toBe("float32");

    // ReLU annotation
    expect(result.annotations.has(reluId)).toBe(true);
    const reluAnn = result.annotations.get(reluId)!;
    expect(reluAnn.inputType!.shape).toHaveLength(2);
    expect(reluAnn.inputType!.shape[0]).toEqual({
      kind: "symbolic",
      name: "B",
    });
    expect(reluAnn.inputType!.shape[1]).toEqual({
      kind: "const",
      value: 784,
    });
    // ReLU preserves both shape and dtype
    expect(reluAnn.outputType.shape).toHaveLength(2);
    expectOutputShape(result, reluId, ["$B", "784"]);
    expect(reluAnn.outputType.dtype).toBe("float32");
  });

  it("1.3: Verify dtypes propagate correctly through chain", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    const reluStereo = d.stereotypes.find((s) => s.name === "ReLU")!;
    d.addModule(reluStereo, 200, 0);
    const reluId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, reluId));

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);

    // Input produces float32 (from Input.json dtype.output)
    expect(result.annotations.get(inputId)!.outputType.dtype).toBe("float32");
    // ReLU preserves dtype from input (no explicit dtype.output in ReLU.json)
    expect(result.annotations.get(reluId)!.outputType.dtype).toBe("float32");
    // ReLU input dtype matches ReLU output dtype
    expect(result.annotations.get(reluId)!.inputType!.dtype).toBe("float32");
  });

  it.skip(
    "1.1: Input → Linear (matching params) — Linear wildcard not at last position (Phase 1 limitation)",
    () => {
      // Linear pattern [B, *, in_features] has wildcard at index 1, not at
      // pattern.length-1.  Phase 1 only supports wildcard as the last element.
    },
  );
});

// ---------------------------------------------------------------------------
// Group 2 — Shape Mismatch Errors
// ---------------------------------------------------------------------------

describe("TypeEngine — Shape Mismatch Errors", () => {
  it("2.3: Unresolved param ('Undefined') produces no hard error", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    // Input output pattern is [B, out_features].  If out_features is
    // "Undefined", resolveParamRef returns undefined, and the dim is
    // treated as symbolic ?out_features — not an error.
    d.updateModule(inputId, {
      params: { out_features: { value: "Undefined" } },
    });

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);
    expect(result.annotations.has(inputId)).toBe(true);

    const ann = result.annotations.get(inputId)!;
    expect(ann.outputType.shape[0]).toEqual({ kind: "symbolic", name: "B" });
    // Second dimension should be symbolic because out_features was undefined
    expect(ann.outputType.shape[1].kind).toBe("symbolic");
  });

  it.skip(
    "2.1: Linear in_features mismatch — same wildcard position issue",
    () => {},
  );

  it.skip(
    "2.2: Extra dimensions not covered — needs fixed-dim pattern",
    () => {},
  );

  it.skip(
    "2.4: Dtype mismatch — dtype constraints not enforced in Phase 1",
    () => {},
  );
});

// ---------------------------------------------------------------------------
// Group 3 — Edge Cases
// ---------------------------------------------------------------------------

describe("TypeEngine — Edge Cases", () => {
  it("3.1: No type_signature (Fork) produces warning, not error", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    const forkStereo = d.stereotypes.find((s) => s.name === "Fork")!;
    d.addModule(forkStereo, 200, 0);
    const forkId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, forkId));

    const result = TypeEngine.infer(d);
    // No hard errors — just a warning
    expectTypeSuccess(result);

    // Warning about missing type signature
    const forkWarnings = result.errors.filter(
      (e) => e.severity === "warning" && e.nodeId === forkId,
    );
    expect(forkWarnings.length).toBeGreaterThan(0);
    expect(forkWarnings[0].message).toContain("No type signature");

    // Fork output type is unknown placeholder
    expect(result.annotations.get(forkId)!.outputType.dtype).toBe("unknown");
    expect(result.annotations.get(forkId)!.outputType.shape).toEqual([]);
  });

  it("3.2: Disconnected node (floating) is not traversed", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    // Connected ReLU
    const reluStereo = d.stereotypes.find((s) => s.name === "ReLU")!;
    d.addModule(reluStereo, 200, 0);
    const reluId = d.nodes[1].id;

    // Floating ReLU (has type_signature but no incoming edges → silently skipped)
    d.addModule(reluStereo, 400, 0);
    const floatId = d.nodes[2].id;

    d.edges.push(edge("e1", inputId, reluId));

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);

    // Connected path annotated
    expect(result.annotations.has(inputId)).toBe(true);
    expect(result.annotations.has(reluId)).toBe(true);

    // Floating node with no incoming edges is silently skipped
    expect(result.annotations.has(floatId)).toBe(false);
  });

  it("3.3: Empty diagram (only Input node) works fine", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);
    expect(result.annotations.has(inputId)).toBe(true);

    const ann = result.annotations.get(inputId)!;
    expect(ann.inputType).toBeUndefined();
    expect(ann.outputType.shape).toHaveLength(2);
    expect(ann.outputType.shape[1]).toEqual({ kind: "const", value: 784 });
    expect(ann.outputType.dtype).toBe("float32");
  });

  it("3.4: Join node (no type_signature) emits warning", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "10" } } });

    // Addition join has no type_signature field in its JSON
    const addStereo = d.stereotypes.find((s) => s.name === "Addition")!;
    d.addJoinNode(addStereo, 200, 0);
    const joinId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, joinId));

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);

    // Warning about no type signature (since Addition has no type_signature)
    const joinWarnings = result.errors.filter(
      (e) => e.severity === "warning" && e.nodeId === joinId,
    );
    expect(joinWarnings.length).toBeGreaterThan(0);
  });

  it.skip(
    "3.5: Multiple symbolic dimensions binding consistently (Phase 2+)",
    () => {},
  );
});

// ---------------------------------------------------------------------------
// Group 4 — Wildcard Behavior
// ---------------------------------------------------------------------------

describe("TypeEngine — Wildcard Behavior", () => {
  it("Wildcard preserves input shape through ReLU", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    // ReLU pattern is [*] — wildcard captures all incoming dims
    const reluStereo = d.stereotypes.find((s) => s.name === "ReLU")!;
    d.addModule(reluStereo, 200, 0);
    const reluId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, reluId));

    const result = TypeEngine.infer(d);
    expectTypeSuccess(result);

    // Wildcard [*] captures both [B, 784] dims and re-emits them
    const reluAnn = result.annotations.get(reluId)!;
    expect(reluAnn.outputType.shape).toHaveLength(2);
    expectOutputShape(result, reluId, ["$B", "784"]);
    expect(reluAnn.outputType.dtype).toBe("float32");
  });

  it.skip(
    "4.1: Wildcard consumes zero dimensions — depends on Linear pattern fix",
    () => {},
  );

  it.skip(
    "4.2: Wildcard consumes one intermediate dimension (Phase 2+)",
    () => {},
  );
});

// ---------------------------------------------------------------------------
// Group 5 — Error Message Quality
// ---------------------------------------------------------------------------

describe("TypeEngine — Error Message Quality", () => {
  it("5.1: Error/warning includes nodeId", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    const forkStereo = d.stereotypes.find((s) => s.name === "Fork")!;
    d.addModule(forkStereo, 200, 0);
    const forkId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, forkId));

    const result = TypeEngine.infer(d);
    const forkErrors = result.errors.filter((e) => e.nodeId === forkId);
    expect(forkErrors.length).toBeGreaterThan(0);
    // The nodeId must be a non-empty string
    expect(forkErrors[0].nodeId).toBe(forkId);
    expect(typeof forkErrors[0].nodeId).toBe("string");
    expect(forkErrors[0].nodeId.length).toBeGreaterThan(0);
  });

  it("5.2: Error message is human-readable (no stack traces, > 10 chars)", () => {
    const d = new Diagram();
    const inputId = d.nodes[0].id;
    d.updateModule(inputId, { params: { out_features: { value: "784" } } });

    const forkStereo = d.stereotypes.find((s) => s.name === "Fork")!;
    d.addModule(forkStereo, 200, 0);
    const forkId = d.nodes[1].id;
    d.edges.push(edge("e1", inputId, forkId));

    const result = TypeEngine.infer(d);
    const forkErrors = result.errors.filter((e) => e.nodeId === forkId);
    expect(forkErrors.length).toBeGreaterThan(0);

    // Human-readable: more than 10 characters
    expect(forkErrors[0].message.length).toBeGreaterThan(10);
    // Should not contain stack traces or internal names
    expect(forkErrors[0].message).not.toContain("at ");
    expect(forkErrors[0].message).not.toContain("Error:");
    expect(forkErrors[0].message).not.toContain("TypeError");
  });
});
