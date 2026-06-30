import type { Edge } from "@xyflow/svelte";

export interface ConnectionValidation {
  valid: boolean;
  reason?: string;
}

export interface ValidationError {
  code: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
}

export interface ValidationWarning {
  code: string;
  message: string;
  nodeId?: string;
}

export interface GraphValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

/**
 * Check if a connection is valid based on target handle availability.
 * Extracted from utils.ts:checkValidConnection — operates on plain Edge[],
 * not a Diagram instance.
 */
export function checkValidConnection(
  edges: Edge[],
  source: string,
  target: string,
  sourceHandle?: string,
  targetHandle?: string
): ConnectionValidation {
  // Self-loop check
  if (source === target) {
    return { valid: false, reason: "Cannot connect a node to itself" };
  }

  // Target handle occupancy check
  const isTargetTaken = edges.some(
    (e) => e.target === target && e.targetHandle === targetHandle
  );

  if (isTargetTaken) {
    return { valid: false, reason: `Target handle '${targetHandle}' on node '${target}' is already occupied` };
  }

  return { valid: true };
}

// Future: validateGraph, validateConnections, validateParameters, validateSubflows
// will be added here in Phase 2 when the MCP server needs them.
// For Phase 1, only checkValidConnection is extracted.
