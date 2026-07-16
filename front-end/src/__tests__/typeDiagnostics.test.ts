/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 */

import { describe, expect, it } from "vitest";
import {
  getNodeDiagnosticSummary,
  getNodeTypeInfo,
  serializeTypeResult,
} from "../conversion/typeDiagnostics";
import type { TypeResult } from "../conversion/tensortypes";

function result(overrides: Partial<TypeResult> = {}): TypeResult {
  return {
    ok: true,
    annotations: new Map([
      [
        "n1",
        {
          nodeId: "n1",
          outputType: {
            shape: [{ kind: "const", value: 10 }],
            dtype: "float32",
          },
        },
      ],
    ]),
    errors: [],
    warnings: [],
    suggestions: [],
    ...overrides,
  };
}

describe("type diagnostics view model", () => {
  it("serializes annotation maps for BrowserRPC/MCP", () => {
    const serialized = serializeTypeResult(result());
    expect(serialized.annotations.n1.outputType.shape).toEqual([
      { kind: "const", value: 10 },
    ]);
  });

  it("filters diagnostics for one node", () => {
    const info = getNodeTypeInfo(
      result({
        warnings: [{ nodeId: "n1", message: "dtype warning", kind: "dtype" }],
        suggestions: [{
          nodeId: "n2",
          param: "in_features",
          value: 10,
          reason: "input dimension",
        }],
      }),
      "n1",
    );

    expect(info.annotation?.nodeId).toBe("n1");
    expect(info.warnings).toHaveLength(1);
    expect(info.suggestions).toHaveLength(0);
  });

  it("prioritizes errors over warnings and suggestions for node badges", () => {
    const summary = getNodeDiagnosticSummary(
      result({
        ok: false,
        errors: [{ nodeId: "n1", message: "hard error", severity: "error" }],
        warnings: [{ nodeId: "n1", message: "warning", kind: "shape" }],
        suggestions: [{
          nodeId: "n1",
          param: "in_features",
          value: 10,
          reason: "input dimension",
        }],
      }),
      "n1",
    );

    expect(summary).toEqual({ severity: "error", message: "hard error" });
  });
});
