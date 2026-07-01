/**
 * Diagram Inspection Tools — read-only queries of the diagram state.
 *
 * These tools allow the LLM to inspect the current graph structure, query
 * individual nodes/edges, compute statistics, and list available stereotypes.
 * None of these tools mutate the diagram or push history snapshots.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { Node, Edge } from "@nnmodelling/front-end/core/types";
import { NodeNotFoundError } from "../errors";

// ── Statistics Helpers ─────────────────────────────────────────────────

/**
 * Compute the longest path depth in the graph using BFS from all input nodes.
 * Returns 0 if there are no input nodes or the graph is empty.
 * Uses stereotype-based input detection for consistency with graph_statistics.
 */
function computeMaxDepth(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (name: string) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  const adjacency = new Map<string, string[]>();
  for (const edge of graph.edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    adjacency.get(edge.source)!.push(edge.target);
  }

  const inputs = graph.nodes.filter((n) => {
    const stereoName = (n.data as Record<string, unknown> | undefined)?.stereotype as string | undefined;
    if (!stereoName) return false;
    const stereo = getStereotype(stereoName);
    return stereo?.isInput === true;
  });
  if (inputs.length === 0) return 0;

  let maxDepth = 0;
  for (const input of inputs) {
    const visited = new Set<string>();
    const queue: Array<{ id: string; depth: number }> = [{ id: input.id, depth: 0 }];
    visited.add(input.id);

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      maxDepth = Math.max(maxDepth, depth);

      for (const childId of adjacency.get(id) ?? []) {
        if (!visited.has(childId)) {
          visited.add(childId);
          queue.push({ id: childId, depth: depth + 1 });
        }
      }
    }
  }

  return maxDepth;
}

/**
 * Compute the average number of outgoing edges per non-loss node.
 * Loss/output nodes are excluded since they typically have no outgoing edges.
 * Uses stereotype-based loss detection for consistency with graph_statistics.
 */
function computeAvgFanOut(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (name: string) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  let totalOut = 0;
  let count = 0;

  for (const node of graph.nodes) {
    const stereoName = (node.data as Record<string, unknown> | undefined)?.stereotype as string | undefined;
    if (stereoName) {
      const stereo = getStereotype(stereoName);
      if (stereo?.isLoss) continue;
    }

    totalOut += graph.edges.filter((e) => e.source === node.id).length;
    count++;
  }

  return count > 0 ? Math.round((totalOut / count) * 100) / 100 : 0;
}

/**
 * Detect whether the graph contains any cycles using Kahn's algorithm.
 * Returns true if the graph is acyclic (DAG).
 */
function isCycleFree(diagram: { nodes: Node[]; edges: Edge[] }): boolean {
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const node of diagram.nodes) {
    inDegree.set(node.id, 0);
    adjacency.set(node.id, []);
  }

  for (const edge of diagram.edges) {
    adjacency.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  let visitedCount = 0;
  while (queue.length > 0) {
    const id = queue.shift()!;
    visitedCount++;

    for (const neighbor of adjacency.get(id) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  return visitedCount === diagram.nodes.length;
}

// ── Tools ──────────────────────────────────────────────────────────────

export const get_graph = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    return {
      nodes: ctx.diagram.nodes,
      edges: ctx.diagram.edges,
    };
  },
};

export const get_node = {
  schema: z.object({
    nodeId: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<Node> {
    const node = ctx.diagram.getNodeById(input.nodeId);
    if (!node) throw new NodeNotFoundError(input.nodeId);
    return node;
  },
};

export const get_edges = {
  schema: z.object({
    nodeId: z.string().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ edges: Edge[] }> {
    if (input.nodeId) {
      // Validate that the node exists
      const node = ctx.diagram.getNodeById(input.nodeId);
      if (!node) throw new NodeNotFoundError(input.nodeId);

      // Return only edges connected to the specified node
      const edges = ctx.diagram.edges.filter(
        (e) => e.source === input.nodeId || e.target === input.nodeId
      );
      return { edges };
    }

    // Return all edges
    return { edges: ctx.diagram.edges };
  },
};

export const get_subflow = {
  schema: z.object({
    parentId: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ nodes: Node[]; edges: Edge[] }> {
    // Validate the parent subflow node exists
    const parent = ctx.diagram.getNodeById(input.parentId);
    if (!parent) throw new NodeNotFoundError(input.parentId);

    // Collect all direct children of this subflow
    const subflowNodes = ctx.diagram.nodes.filter(
      (n) => n.parentId === input.parentId
    );
    const subflowNodeIds = new Set(subflowNodes.map((n) => n.id));

    // Collect edges where both endpoints are inside the subflow
    const subflowEdges = ctx.diagram.edges.filter(
      (e) => subflowNodeIds.has(e.source) && subflowNodeIds.has(e.target)
    );

    return { nodes: subflowNodes, edges: subflowEdges };
  },
};

export const graph_statistics = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    nodeCount: number;
    edgeCount: number;
    moduleCount: number;
    joinCount: number;
    subflowCount: number;
    inputCount: number;
    lossCount: number;
    maxDepth: number;
    avgFanOut: number;
    cycleFree: boolean;
  }> {
    const { nodes, edges } = ctx.diagram;

    let moduleCount = 0;
    let joinCount = 0;
    let subflowCount = 0;
    let inputCount = 0;
    let lossCount = 0;

    const getStereo = (name: string) => ctx.diagram.getStereotype(name);

    for (const node of nodes) {
      if (node.type === "join") {
        joinCount++;
      } else if (node.type === "subflow") {
        subflowCount++;
      } else {
        moduleCount++;
      }

      const stereoName = (node.data as Record<string, unknown> | undefined)?.stereotype as string | undefined;
      if (stereoName) {
        const stereo = getStereo(stereoName);
        if (stereo?.isInput) inputCount++;
        if (stereo?.isLoss) lossCount++;
      }
    }

    return {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      moduleCount,
      joinCount,
      subflowCount,
      inputCount,
      lossCount,
      maxDepth: computeMaxDepth(ctx.diagram, getStereo),
      avgFanOut: computeAvgFanOut(ctx.diagram, getStereo),
      cycleFree: isCycleFree(ctx.diagram),
    };
  },
};

export const list_stereotypes = {
  schema: z.object({
    category: z.string().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    stereotypes: Array<{
      name: string;
      category: string;
      pythonClassName: string;
      isJoin: boolean;
      isInput: boolean;
      isLoss: boolean;
      isSubFlow: boolean;
      parameters: Record<
        string,
        { type: string; default: string; position?: string }
      >;
    }>;
  }> {
    // Safety check: stereotypes may not be initialized yet
    const allStereotypes = ctx.diagram.stereotypes ?? [];

    const filtered = input.category
      ? allStereotypes.filter((s) => s.category === input.category)
      : allStereotypes;

    return {
      stereotypes: filtered.map((s) => ({
        name: s.name,
        category: s.category,
        pythonClassName: s.pythonClassName,
        isJoin: s.isJoin,
        isInput: s.isInput,
        isLoss: s.isLoss,
        isSubFlow: s.isSubFlow,
        parameters: s.parameters,
      })),
    };
  },
};
