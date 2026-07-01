/**
 * Selection Tools — select, clear, query, and select-all operations on
 * diagram nodes and edges.
 *
 * Selection state is stored on each node/edge via the `selected` boolean
 * field. The DiagramCore emits a `selection_changed` DomainEvent whenever
 * selection state is modified.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { Node, Edge } from "@nnmodelling/front-end/core/types";
import { NodeNotFoundError } from "../errors";

// ── Schemas ────────────────────────────────────────────────────────────

export const select_nodes = {
  schema: z.object({
    nodeIds: z.array(z.string()),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ selectedNodeIds: string[]; selectedEdgeIds: string[] }> {
    const { nodeIds } = input;

    // Validate that all requested nodes exist
    for (const id of nodeIds) {
      if (!ctx.diagram.getNodeById(id)) throw new NodeNotFoundError(id);
    }

    ctx.diagram.selectNodes(nodeIds);

    return {
      selectedNodeIds: nodeIds,
      selectedEdgeIds: ctx.diagram.getSelectedEdges().map((e) => e.id),
    };
  },
};

export const clear_selection = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{ cleared: boolean }> {
    ctx.diagram.clearSelection();
    return { cleared: true };
  },
};

export const get_selection = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    nodeIds: string[];
    edgeIds: string[];
    nodes: Node[];
    edges: Edge[];
  }> {
    const selectedNodes = ctx.diagram.getSelectedNodes();
    const selectedEdges = ctx.diagram.getSelectedEdges();

    return {
      nodeIds: selectedNodes.map((n) => n.id),
      edgeIds: selectedEdges.map((e) => e.id),
      nodes: selectedNodes,
      edges: selectedEdges,
    };
  },
};

export const select_all = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{ nodeCount: number }> {
    const allNodeIds = ctx.diagram.nodes.map((n) => n.id);
    ctx.diagram.selectNodes(allNodeIds);

    return { nodeCount: allNodeIds.length };
  },
};
