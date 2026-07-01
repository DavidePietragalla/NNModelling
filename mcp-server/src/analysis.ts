/**
 * Shared Analysis Helpers — graph statistics and validation primitives.
 *
 * These functions are used by both resources (read-only views) and tools
 * (MCP callable handlers) to avoid code duplication. They operate purely
 * on graph data structures with no side effects.
 *
 * @module analysis
 */

import type { Node, Edge } from "@nnmodelling/front-end/core/types";

// ── Helpers ──────────────────────────────────────────────────────────────

/** Extract the stereotype name from a node's data field. */
function getStereoName(node: Node): string | undefined {
  return (node.data as Record<string, unknown> | undefined)?.stereotype as
    | string
    | undefined;
}

// ── Statistics ───────────────────────────────────────────────────────────

/**
 * Compute the longest path depth in the graph using BFS from all input nodes.
 * Returns 0 if there are no input nodes or the graph is empty.
 */
export function computeMaxDepth(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (
    name: string,
  ) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  const adjacency = new Map<string, string[]>();
  for (const edge of graph.edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    adjacency.get(edge.source)!.push(edge.target);
  }

  const inputs = graph.nodes.filter((n) => {
    const stereoName = getStereoName(n);
    if (!stereoName) return false;
    const stereo = getStereotype(stereoName);
    return stereo?.isInput === true;
  });
  if (inputs.length === 0) return 0;

  let maxDepth = 0;
  for (const input of inputs) {
    const visited = new Set<string>();
    const queue: Array<{ id: string; depth: number }> = [
      { id: input.id, depth: 0 },
    ];
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
 */
export function computeAvgFanOut(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (
    name: string,
  ) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  let totalOut = 0;
  let count = 0;

  for (const node of graph.nodes) {
    const stereoName = getStereoName(node);
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
export function isCycleFree(graph: { nodes: Node[]; edges: Edge[] }): boolean {
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const node of graph.nodes) {
    inDegree.set(node.id, 0);
    adjacency.set(node.id, []);
  }

  for (const edge of graph.edges) {
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

  return visitedCount === graph.nodes.length;
}
