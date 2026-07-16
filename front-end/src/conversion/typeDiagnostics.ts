/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 */

import type {
  NodeTypeAnnotation,
  TypeError,
  TypeResult,
  TypeSuggestion,
  TypeWarning,
} from "./tensortypes";

/** JSON-safe representation of a full type-inference result. */
export interface SerializedTypeResult {
  ok: boolean;
  annotations: Record<string, NodeTypeAnnotation>;
  errors: TypeError[];
  warnings: TypeWarning[];
  suggestions: TypeSuggestion[];
}

/** Type information and diagnostics associated with one diagram node. */
export interface NodeTypeInfo {
  annotation: NodeTypeAnnotation | null;
  errors: TypeError[];
  warnings: TypeWarning[];
  suggestions: TypeSuggestion[];
}

export interface NodeDiagnosticSummary {
  severity: "error" | "warning" | "suggestion";
  message: string;
}

/** Convert Map-based inference state to a representation suitable for RPC. */
export function serializeTypeResult(result: TypeResult): SerializedTypeResult {
  return {
    ok: result.ok,
    annotations: Object.fromEntries(result.annotations),
    errors: result.errors,
    warnings: result.warnings,
    suggestions: result.suggestions,
  };
}

/** Extract the inferred annotation and diagnostics for a single node. */
export function getNodeTypeInfo(result: TypeResult, nodeId: string): NodeTypeInfo {
  return {
    annotation: result.annotations.get(nodeId) ?? null,
    errors: result.errors.filter((error) => error.nodeId === nodeId),
    warnings: result.warnings.filter((warning) => warning.nodeId === nodeId),
    suggestions: result.suggestions.filter((suggestion) => suggestion.nodeId === nodeId),
  };
}

/** Build the highest-priority badge shown on a diagram node. */
export function getNodeDiagnosticSummary(
  result: TypeResult | null,
  nodeId: string,
): NodeDiagnosticSummary | null {
  if (!result) return null;

  const hardErrors = result.errors.filter(
    (error) => error.nodeId === nodeId && error.severity === "error",
  );
  if (hardErrors.length > 0) {
    return {
      severity: "error",
      message: hardErrors.map((error) => error.message).join("\n"),
    };
  }

  const softMessages = [
    ...result.errors
      .filter((error) => error.nodeId === nodeId)
      .map((error) => error.message),
    ...result.warnings
      .filter((warning) => warning.nodeId === nodeId)
      .map((warning) => warning.message),
  ];
  if (softMessages.length > 0) {
    return { severity: "warning", message: softMessages.join("\n") };
  }

  const suggestions = result.suggestions.filter(
    (suggestion) => suggestion.nodeId === nodeId,
  );
  if (suggestions.length > 0) {
    return {
      severity: "suggestion",
      message: suggestions
        .map((suggestion) => `Set ${suggestion.param}=${suggestion.value}: ${suggestion.reason}`)
        .join("\n"),
    };
  }

  return null;
}
