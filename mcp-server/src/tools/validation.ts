/**
 * Graph Validation Tools — validate diagram structure, connections, parameters,
 * and subflow hierarchy.
 *
 * These tools run static analysis on the current DiagramCore state and return
 * structured error/warning reports. They do NOT mutate the diagram.
 *
 * Validation checks include:
 *   - Graph structure (cycles, disconnected nodes, empty graph)
 *   - Connection rules (self-loops, target handle occupancy)
 *   - Parameter types (against stereotype definitions)
 *   - Subflow integrity (entry node, internal cycles, orphaned nodes)
 *
 * @module tools/validation
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { Node, Edge } from "@nnmodelling/front-end/core/types";

// ── Types ───────────────────────────────────────────────────────────────

/**
 * A single validation issue (error or warning).
 */
interface ValidationIssue {
  code: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────

/**
 * Run Kahn's topological sort to detect cycles in a subgraph.
 *
 * @param nodeIds - Set of node IDs in the subgraph.
 * @param edges   - Edges within the subgraph.
 * @returns True if the subgraph contains a cycle.
 */
function hasCycle(nodeIds: Set<string>, edges: Edge[]): boolean {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  for (const id of nodeIds) {
    inDegree.set(id, 0);
    adj.set(id, []);
  }

  for (const e of edges) {
    if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) continue;
    adj.get(e.source)?.push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, degree] of inDegree) {
    if (degree === 0) queue.push(id);
  }

  const sorted = new Set<string>();
  while (queue.length > 0) {
    const id = queue.shift()!;
    sorted.add(id);
    for (const target of adj.get(id) || []) {
      const d = (inDegree.get(target) || 1) - 1;
      inDegree.set(target, d);
      if (d === 0) queue.push(target);
    }
  }

  return sorted.size !== nodeIds.size;
}

/**
 * Get the stereotype name from a node's data field.
 */
function getStereotypeName(node: Node): string | undefined {
  return (node.data as Record<string, unknown>)?.stereotype as string | undefined;
}

/**
 * Compute the subflow nesting depth for a given node.
 * Top-level subflow = depth 1, nested inside another subflow = depth 2, etc.
 */
function getSubflowDepth(node: Node, allNodes: Node[]): number {
  let depth = 1;
  let currentId = node.parentId;
  while (currentId) {
    const parent = allNodes.find((n) => n.id === currentId);
    if (!parent || parent.type !== "subflow") break;
    depth++;
    currentId = parent.parentId;
  }
  return depth;
}

/**
 * Get the parameter value type from a stereotype definition by parameter key.
 * Returns "string" if the stereotype or parameter is not found.
 */
function getParamType(
  ctx: ServerContext,
  node: Node,
  key: string,
): string | undefined {
  const stereoName = getStereotypeName(node);
  if (!stereoName) return undefined;

  const stereo = ctx.diagram.getStereotype(stereoName);
  if (!stereo) return undefined;

  const paramDef = stereo.parameters[key];
  return paramDef?.type;
}

// ── validate_graph ─────────────────────────────────────────────────────

export const validate_graph = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    valid: boolean;
    errors: ValidationIssue[];
    warnings: ValidationIssue[];
  }> {
    const errors: ValidationIssue[] = [];
    const warnings: ValidationIssue[] = [];

    const { nodes, edges } = ctx.diagram;

    // ── 1. Empty graph ──────────────────────────────
    if (nodes.length === 0) {
      errors.push({
        code: "EMPTY_GRAPH",
        message: "The diagram is empty. Add at least one Input node and one Loss node.",
      });
      return { valid: false, errors, warnings };
    }

    // ── 2. Input node count ─────────────────────────
    const inputNodeIds = new Set<string>();
    for (const n of nodes) {
      const stereoName = getStereotypeName(n);
      if (stereoName) {
        const stereo = ctx.diagram.getStereotype(stereoName);
        if (stereo?.isInput) inputNodeIds.add(n.id);
      } else if (n.data && (n.data as Record<string, unknown>).stereotype === "Input") {
        inputNodeIds.add(n.id);
      }
    }

    if (inputNodeIds.size === 0) {
      errors.push({
        code: "MISSING_INPUT",
        message: "No Input node found. Every diagram must have exactly one Input node.",
      });
    } else if (inputNodeIds.size > 1) {
      errors.push({
        code: "MULTIPLE_INPUTS",
        message: `Found ${inputNodeIds.size} Input nodes. Every diagram must have exactly one Input node.`,
      });
    }

    // ── 3. Loss node presence ───────────────────────
    // A loss node is a node with no outgoing edges (terminal node).
    const nodesWithOutgoing = new Set(edges.map((e) => e.source));
    const terminalNodes = nodes.filter((n) => !nodesWithOutgoing.has(n.id));

    const lossNodeIds: string[] = [];
    for (const n of terminalNodes) {
      const stereoName = getStereotypeName(n);
      if (stereoName) {
        const stereo = ctx.diagram.getStereotype(stereoName);
        if (stereo?.isLoss) lossNodeIds.push(n.id);
      } else if (
        n.data &&
        typeof (n.data as Record<string, unknown>).stereotype === "string" &&
        String((n.data as Record<string, unknown>).stereotype).endsWith("Loss")
      ) {
        lossNodeIds.push(n.id);
      }
    }

    if (lossNodeIds.length === 0) {
      warnings.push({
        code: "NO_LOSS_NODE",
        message:
          "No Loss node (terminal node with Loss category) detected. " +
          "The diagram will compile but may not produce training metrics. " +
          "Add a loss node (e.g. CrossEntropyLoss, MSELoss) as the final output.",
      });
    }

    // ── 4. Disconnected / orphan nodes ──────────────
    const nodesWithIncoming = new Set(edges.map((e) => e.target));

    for (const n of nodes) {
      const stereoName = getStereotypeName(n);
      if (stereoName) {
        const stereo = ctx.diagram.getStereotype(stereoName);
        if (stereo?.isInput) continue; // Input nodes naturally have no incoming edges
        if (stereo?.isLoss) continue;  // Loss nodes naturally have no outgoing edges
      }

      const hasIncoming = nodesWithIncoming.has(n.id);
      const hasOutgoing = nodesWithOutgoing.has(n.id);

      if (!hasIncoming && !hasOutgoing) {
        warnings.push({
          code: "ORPHAN_NODE",
          message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no incoming or outgoing connections.`,
          nodeId: n.id,
        });
      } else if (!hasIncoming) {
        warnings.push({
          code: "DISCONNECTED_INPUT",
          message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no incoming connections.`,
          nodeId: n.id,
        });
      } else if (!hasOutgoing) {
        const stereoName_ = getStereotypeName(n);
        if (!stereoName_ || !ctx.diagram.getStereotype(stereoName_)?.isLoss) {
          warnings.push({
            code: "DISCONNECTED_OUTPUT",
            message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no outgoing connections but is not a Loss node.`,
            nodeId: n.id,
          });
        }
      }
    }

    // ── 5. Cycle detection ──────────────────────────
    const allNodeIds = new Set(nodes.map((n) => n.id));
    if (hasCycle(allNodeIds, edges)) {
      errors.push({
        code: "GRAPH_CYCLE",
        message:
          "The diagram contains one or more cycles. " +
          "Cycle detection uses Kahn's algorithm on the full edge set. " +
          "Cycles are not allowed in the computation graph.",
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  },
};

// ── validate_connections ──────────────────────────────────────────────

export const validate_connections = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    valid: boolean;
    errors: ValidationIssue[];
  }> {
    const errors: ValidationIssue[] = [];
    const { edges } = ctx.diagram;

    // ── 1. Self-loops ───────────────────────────────
    for (const e of edges) {
      if (e.source === e.target) {
        errors.push({
          code: "SELF_LOOP",
          message: `Edge connects node "${e.source}" to itself. Remove the self-loop.`,
          edgeId: e.id,
          nodeId: e.source,
        });
      }
    }

    // ── 2. Target handle occupancy ──────────────────
    // Each target handle should have at most one incoming edge.
    // Check plain targets without handles — each node can have only one
    // connection to its "in" target (default handle).
    const targetMap = new Map<string, Edge[]>();
    for (const e of edges) {
      const key = e.targetHandle && e.targetHandle !== "in"
        ? `${e.target}:${e.targetHandle}`
        : e.target;
      if (!targetMap.has(key)) targetMap.set(key, []);
      targetMap.get(key)!.push(e);
    }

    for (const [key, connectedEdges] of targetMap) {
      if (connectedEdges.length > 1) {
        const [targetId] = key.split(":");
        errors.push({
          code: "TARGET_HANDLE_OCCUPIED",
          message:
            `Node "${targetId}" (handle "${key.includes(":") ? key.split(":")[1] : "in"}") ` +
            `has ${connectedEdges.length} incoming connections. Each target handle accepts at most one edge.`,
          nodeId: targetId,
          edgeId: connectedEdges[1].id, // Point to the duplicate edge
        });
      }
    }

    // ── 3. Invalid source/target node references ────
    const nodeIds = new Set(ctx.diagram.nodes.map((n) => n.id));
    for (const e of edges) {
      if (!nodeIds.has(e.source)) {
        errors.push({
          code: "INVALID_SOURCE",
          message: `Edge references non-existent source node "${e.source}".`,
          edgeId: e.id,
        });
      }
      if (!nodeIds.has(e.target)) {
        errors.push({
          code: "INVALID_TARGET",
          message: `Edge references non-existent target node "${e.target}".`,
          edgeId: e.id,
        });
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  },
};

// ── validate_parameters ───────────────────────────────────────────────

export const validate_parameters = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    valid: boolean;
    errors: ValidationIssue[];
  }> {
    const errors: ValidationIssue[] = [];

    for (const node of ctx.diagram.nodes) {
      const stereoName = getStereotypeName(node);
      if (!stereoName) continue;

      const stereo = ctx.diagram.getStereotype(stereoName);
      if (!stereo) continue;

      const nodeParams: Record<string, unknown> =
        ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

      // Check each param defined in the stereotype
      for (const [key, paramDef] of Object.entries(stereo.parameters)) {
        // Skip parameters whose default is "Undefined" — they might be set by the user
        const hasValue = key in nodeParams;
        const rawValue = nodeParams[key];

        if (!hasValue || rawValue === undefined || rawValue === null || rawValue === "") {
          // Only error if the default is "Undefined" (meaning it's required)
          if (paramDef.default === "Undefined" || paramDef.default === "") {
            errors.push({
              code: "MISSING_PARAMETER",
              message:
                `Node "${(node.data as Record<string, unknown>)?.name ?? node.id}" ` +
                `(stereotype: ${stereoName}) is missing required parameter "${key}".`,
              nodeId: node.id,
            });
          }
          continue;
        }

        // Type validation
        const valueStr = String(rawValue);
        const expectedType = paramDef.type?.toLowerCase() ?? "any";

        switch (expectedType) {
          case "int":
          case "integer": {
            const intVal = parseInt(valueStr, 10);
            if (isNaN(intVal) || String(intVal) !== valueStr.trim()) {
              errors.push({
                code: "PARAMETER_TYPE_MISMATCH",
                message:
                  `Parameter "${key}" on node "${(node.data as Record<string, unknown>)?.name ?? node.id}" ` +
                  `expected type "int", got "${valueStr}".`,
                nodeId: node.id,
              });
            }
            break;
          }
          case "float":
          case "number": {
            const floatVal = parseFloat(valueStr);
            if (isNaN(floatVal)) {
              errors.push({
                code: "PARAMETER_TYPE_MISMATCH",
                message:
                  `Parameter "${key}" on node "${(node.data as Record<string, unknown>)?.name ?? node.id}" ` +
                  `expected type "float", got "${valueStr}".`,
                nodeId: node.id,
              });
            }
            break;
          }
          case "bool":
          case "boolean": {
            if (valueStr !== "true" && valueStr !== "false" && valueStr !== "True" && valueStr !== "False") {
              errors.push({
                code: "PARAMETER_TYPE_MISMATCH",
                message:
                  `Parameter "${key}" on node "${(node.data as Record<string, unknown>)?.name ?? node.id}" ` +
                  `expected type "bool", got "${valueStr}".`,
                nodeId: node.id,
              });
            }
            break;
          }
          // "str", "string", and unknown types: accept any value
        }
      }

      // Check for unknown parameters (not in stereotype definition)
      const stereoParamKeys = new Set(Object.keys(stereo.parameters));
      for (const key of Object.keys(nodeParams)) {
        if (!stereoParamKeys.has(key)) {
          errors.push({
            code: "UNKNOWN_PARAMETER",
            message:
              `Node "${(node.data as Record<string, unknown>)?.name ?? node.id}" ` +
              `(stereotype: ${stereoName}) has unknown parameter "${key}". ` +
              `This parameter is not defined in the stereotype schema.`,
            nodeId: node.id,
          });
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  },
};

// ── validate_subflows ─────────────────────────────────────────────────

export const validate_subflows = {
  schema: z.object({
    parentId: z.string().optional(),
    maxDepth: z.number().int().positive().default(10),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<{
    valid: boolean;
    errors: ValidationIssue[];
    warnings: ValidationIssue[];
  }> {
    const errors: ValidationIssue[] = [];
    const warnings: ValidationIssue[] = [];

    // Identify all subflow container nodes
    const subflowNodes = input.parentId
      ? [ctx.diagram.getNodeById(input.parentId)].filter(Boolean) as Node[]
      : ctx.diagram.nodes.filter((n) => n.type === "subflow");

    // If a specific parentId was given but no subflow found
    if (input.parentId && subflowNodes.length === 0) {
      errors.push({
        code: "SUBFLOW_NOT_FOUND",
        message: `No subflow node found with ID "${input.parentId}".`,
        nodeId: input.parentId,
      });
      return { valid: false, errors, warnings: [] };
    }

    if (subflowNodes.length === 0) {
      // No subflows at all — this is valid
      return { valid: true, errors: [], warnings: [] };
    }

    for (const subflowNode of subflowNodes) {
      const subflowId = subflowNode.id;
      const subflowName =
        (subflowNode.data as Record<string, unknown>)?.name as string ?? subflowId;

      // Collect internal nodes and edges
      const internalNodes = ctx.diagram.nodes.filter(
        (n) => n.parentId === subflowId,
      );
      const internalNodeIds = new Set(internalNodes.map((n) => n.id));
      const internalEdges = ctx.diagram.edges.filter(
        (e) => internalNodeIds.has(e.source) && internalNodeIds.has(e.target),
      );

      // ── 1. Empty subflow ──────────────────────────
      if (internalNodes.length === 0) {
        errors.push({
          code: "EMPTY_SUBFLOW",
          message: `Subflow "${subflowName}" (${subflowId}) has no internal nodes. Add at least one node inside the subflow.`,
          nodeId: subflowId,
        });
        continue;
      }

      // ── 2. Entry node detection ────────────────────
      // An entry node has no incoming edges from within the subflow.
      const nodesWithIncomingInternal = new Set(internalEdges.map((e) => e.target));
      const entryCandidates = internalNodes.filter(
        (n) => !nodesWithIncomingInternal.has(n.id),
      );

      if (entryCandidates.length === 0) {
        errors.push({
          code: "SUBFLOW_NO_ENTRY",
          message:
            `Subflow "${subflowName}" (${subflowId}) has no entry node. ` +
            `Every node in the subflow has an incoming edge, creating a cycle or closed loop.`,
          nodeId: subflowId,
        });
      }

      // ── 3. Internal cycles ────────────────────────
      if (hasCycle(internalNodeIds, internalEdges)) {
        errors.push({
          code: "SUBFLOW_CYCLE",
          message:
            `Subflow "${subflowName}" (${subflowId}) contains a cycle. ` +
            `The internal computation graph must be acyclic.`,
          nodeId: subflowId,
        });
      }

      // ── 4. Orphaned internal nodes ────────────────
      const nodesWithIncomingOrOutgoing = new Set<string>();
      for (const e of internalEdges) {
        nodesWithIncomingOrOutgoing.add(e.source);
        nodesWithIncomingOrOutgoing.add(e.target);
      }

      for (const n of internalNodes) {
        if (!nodesWithIncomingOrOutgoing.has(n.id)) {
          errors.push({
            code: "SUBFLOW_ORPHAN",
            message:
              `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" ` +
              `inside subflow "${subflowName}" (${subflowId}) is orphaned — ` +
              `it has no connections to other nodes in the subflow.`,
            nodeId: n.id,
          });
        }
      }

      // ── 5. Internal subflow nesting ────────────────────
      // Check nesting depth against maxDepth parameter.
      // Warnings for nested subflows within allowed depth; errors if exceeded.
      const nestedSubflows = internalNodes.filter((n) => n.type === "subflow");
      const maxNestDepth = input.maxDepth;
      for (const nested of nestedSubflows) {
        const depth = getSubflowDepth(nested, ctx.diagram.nodes);
        const nestedName =
          (nested.data as Record<string, unknown>)?.name as string ?? nested.id;

        if (depth > maxNestDepth) {
          errors.push({
            code: "SUBFLOW_DEPTH_EXCEEDED",
            message:
              `Subflow "${subflowName}" (${subflowId}) contains a nested subflow ` +
              `"${nestedName}" (${nested.id}) at depth ${depth}, ` +
              `exceeding the maximum allowed depth of ${maxNestDepth}.`,
            nodeId: nested.id,
          });
        } else {
          warnings.push({
            code: "NESTED_SUBFLOW",
            message:
              `Subflow "${subflowName}" (${subflowId}) contains a nested subflow ` +
              `"${nestedName}" (${nested.id}) at depth ${depth}. ` +
              `Nested subflows are supported but may increase compilation complexity.`,
            nodeId: nested.id,
          });
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  },
};
