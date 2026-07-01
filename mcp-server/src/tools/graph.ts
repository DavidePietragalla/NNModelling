/**
 * Graph Manipulation Tools — create, delete, connect, disconnect, move,
 * duplicate nodes and create subflows.
 *
 * Every mutating tool pushes a history snapshot before making changes.
 * Inputs are validated with Zod schemas; errors use the MCPServerError hierarchy.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { Node } from "@nnmodelling/front-end/core/types";
import {
  StereotypeNotFoundError,
  NodeNotFoundError,
  EdgeNotFoundError,
  InvalidPositionError,
} from "../errors";

// ── Schemas ────────────────────────────────────────────────────────────

export const create_node = {
  schema: z.object({
    stereotype: z.string().min(1),
    position: z.object({
      x: z.number(),
      y: z.number(),
    }),
    config: z
      .object({
        name: z.string().optional(),
        color: z.string().optional(),
        width: z.number().optional(),
        height: z.number().optional(),
        params: z.record(z.string(), z.string()).optional(),
        inputsCount: z.number().int().min(1).optional(),
        parentId: z.string().optional(),
      })
      .optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    nodeId: string;
    name: string;
    type: string;
    stereotype: string;
  }> {
    const { stereotype: stereotypeName, position, config } = input;

    // Find stereotype
    const stereotype = ctx.diagram.getStereotype(stereotypeName);
    if (!stereotype) throw new StereotypeNotFoundError(stereotypeName);

    // Validate position
    if (
      typeof position.x !== "number" ||
      typeof position.y !== "number" ||
      isNaN(position.x) ||
      isNaN(position.y) ||
      !isFinite(position.x) ||
      !isFinite(position.y)
    ) {
      throw new InvalidPositionError(position.x, position.y);
    }

    // Push history snapshot before mutation
    ctx.history.pushSnapshot(`create ${stereotypeName} node`, ctx.diagram);

    const prevCount = ctx.diagram.nodes.length;

    if (stereotype.isJoin) {
      ctx.diagram.addJoinNode(stereotype, position.x, position.y, {
        name: config?.name,
        color: config?.color,
        inputsCount: config?.inputsCount ?? 2,
        params: config?.params ?? {},
      });
    } else {
      ctx.diagram.addModule(stereotype, position.x, position.y, {
        name: config?.name,
        color: config?.color,
        width: config?.width,
        height: config?.height,
        params: config?.params ?? {},
      });
    }

    // Find the newly created node (always appended)
    const created = ctx.diagram.nodes[ctx.diagram.nodes.length - 1];

    return {
      nodeId: created.id,
      name: (created.data as Record<string, unknown>)?.name as string ?? stereotypeName,
      type: created.type ?? "custom",
      stereotype: stereotypeName,
    };
  },
};

export const delete_nodes = {
  schema: z.object({
    nodeIds: z.array(z.string()).min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    deletedNodeIds: string[];
    deletedEdgeIds: string[];
    reparentedNodes: Array<{ nodeId: string; oldParentId: string; newParentId: string | null }>;
  }> {
    const { nodeIds } = input;
    const nodeSet = new Set(nodeIds);

    // Validate all nodes exist
    for (const id of nodeIds) {
      if (!ctx.diagram.getNodeById(id)) throw new NodeNotFoundError(id);
    }

    // Capture edges that will be removed
    const deletedEdgeIds = ctx.diagram.edges
      .filter((e) => nodeSet.has(e.source) || nodeSet.has(e.target))
      .map((e) => e.id);

    // Capture nodes that will be reparented (orphans of deleted subflows)
    const reparentBefore = ctx.diagram.nodes
      .filter((n) => n.parentId && nodeSet.has(n.parentId))
      .map((n) => ({ nodeId: n.id, oldParentId: n.parentId! }));

    ctx.history.pushSnapshot(`delete ${nodeIds.length} nodes`, ctx.diagram);
    ctx.diagram.deleteNodes(nodeIds);

    // After deletion, check new parent IDs for reparented nodes
    const reparentedNodes = reparentBefore.map((r) => {
      const n = ctx.diagram.getNodeById(r.nodeId);
      return {
        nodeId: r.nodeId,
        oldParentId: r.oldParentId,
        newParentId: n?.parentId ?? null,
      };
    });

    return {
      deletedNodeIds: [...nodeIds],
      deletedEdgeIds,
      reparentedNodes,
    };
  },
};

export const connect_nodes = {
  schema: z.object({
    source: z.string().min(1),
    target: z.string().min(1),
    sourceHandle: z.string().optional(),
    targetHandle: z.string().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ edgeId: string; source: string; target: string }> {
    const { source, target, sourceHandle, targetHandle } = input;

    // Validate both nodes exist
    if (!ctx.diagram.getNodeById(source)) throw new NodeNotFoundError(source);
    if (!ctx.diagram.getNodeById(target)) throw new NodeNotFoundError(target);

    ctx.history.pushSnapshot(`connect ${source} -> ${target}`, ctx.diagram);

    const edge = ctx.diagram.addEdge(source, target, sourceHandle, targetHandle);

    return {
      edgeId: edge.id,
      source: edge.source,
      target: edge.target,
    };
  },
};

export const disconnect_nodes = {
  schema: z.object({
    source: z.string().min(1),
    target: z.string().min(1),
    targetHandle: z.string().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ removedEdgeIds: string[] }> {
    const { source, target, targetHandle } = input;

    // Find matching edges before removal
    const matchingEdges = ctx.diagram.edges.filter(
      (e) =>
        e.source === source &&
        e.target === target &&
        (targetHandle === undefined || e.targetHandle === targetHandle)
    );

    if (matchingEdges.length === 0) {
      const desc = targetHandle
        ? `${source} -> ${target} (handle: ${targetHandle})`
        : `${source} -> ${target}`;
      throw new EdgeNotFoundError(desc);
    }

    const removedEdgeIds = matchingEdges.map((e) => e.id);

    ctx.history.pushSnapshot(`disconnect ${source} -> ${target}`, ctx.diagram);
    ctx.diagram.removeEdge(source, target, targetHandle);

    return { removedEdgeIds };
  },
};

export const move_nodes = {
  schema: z.object({
    positions: z
      .array(
        z.object({
          id: z.string(),
          x: z.number(),
          y: z.number(),
        })
      )
      .min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ movedCount: number }> {
    const { positions } = input;

    // Validate all nodes exist
    for (const pos of positions) {
      if (!ctx.diagram.getNodeById(pos.id)) throw new NodeNotFoundError(pos.id);
    }

    ctx.history.pushSnapshot(`move ${positions.length} nodes`, ctx.diagram);
    ctx.diagram.moveNodes(positions);

    return { movedCount: positions.length };
  },
};

export const duplicate_nodes = {
  schema: z.object({
    nodeIds: z.array(z.string()).min(1),
    offset: z
      .object({
        x: z.number(),
        y: z.number(),
      })
      .optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ duplicated: Array<{ originalId: string; newId: string }> }> {
    const { nodeIds } = input;
    const offset = input.offset ?? { x: 50, y: 50 };

    // Validate all originals exist and filter to supported types
    const originals: Node[] = [];
    for (const id of nodeIds) {
      const node = ctx.diagram.getNodeById(id);
      if (!node) throw new NodeNotFoundError(id);
      // Skip subflow nodes (not duplicatable via addModule/addJoinNode)
      if (node.type !== "subflow") {
        originals.push(node);
      }
    }

    ctx.history.pushSnapshot(`duplicate ${originals.length} nodes`, ctx.diagram);

    // Track original → new ID mapping by capturing node before/after each add
    const idMap = new Map<string, string>();
    const duplicated: Array<{ originalId: string; newId: string }> = [];

    // Create copies of each original node, capturing IDs reliably
    for (const node of originals) {
      const stereoName = (node.data as Record<string, unknown>)?.stereotype as string | undefined;
      if (!stereoName) continue;

      const stereo = ctx.diagram.getStereotype(stereoName);
      if (!stereo) continue;

      const newX = node.position.x + offset.x;
      const newY = node.position.y + offset.y;

      // Capture length before creation to detect the new node
      const before = ctx.diagram.nodes.length;

      if (stereo.isJoin) {
        ctx.diagram.addJoinNode(stereo, newX, newY, {
          name: (node.data as Record<string, unknown>)?.name as string | undefined,
          color: (node.data as Record<string, unknown>)?.color as string | undefined,
          inputsCount: (node.data as Record<string, unknown>)?.inputsCount as number | undefined,
          params: (node.data as Record<string, unknown>)?.params as Record<string, unknown> | undefined,
        });
      } else {
        ctx.diagram.addModule(stereo, newX, newY, {
          name: (node.data as Record<string, unknown>)?.name as string | undefined,
          color: (node.data as Record<string, unknown>)?.color as string | undefined,
          width: node.width,
          height: node.height,
          params: (node.data as Record<string, unknown>)?.params as Record<string, unknown> | undefined,
        });
      }

      // The new node is appended; capture its ID
      if (ctx.diagram.nodes.length > before) {
        const newNode = ctx.diagram.nodes[ctx.diagram.nodes.length - 1];
        idMap.set(node.id, newNode.id);
        duplicated.push({ originalId: node.id, newId: newNode.id });
      }
    }

    // Copy edges between duplicated nodes (internal connections)
    const edgesToCopy = ctx.diagram.edges.filter(
      (e) => idMap.has(e.source) && idMap.has(e.target)
    );

    for (const edge of edgesToCopy) {
      ctx.diagram.addEdge(
        idMap.get(edge.source)!,
        idMap.get(edge.target)!,
        edge.sourceHandle ?? undefined,
        edge.targetHandle ?? undefined
      );
    }

    return { duplicated };
  },
};

export const create_subflow = {
  schema: z.object({
    position: z.object({
      x: z.number(),
      y: z.number(),
    }),
    label: z.string().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ nodeId: string; name: string }> {
    const { position, label } = input;

    // Validate position
    if (
      typeof position.x !== "number" ||
      typeof position.y !== "number" ||
      isNaN(position.x) ||
      isNaN(position.y) ||
      !isFinite(position.x) ||
      !isFinite(position.y)
    ) {
      throw new InvalidPositionError(position.x, position.y);
    }

    ctx.history.pushSnapshot(`create subflow`, ctx.diagram);

    // Safer node detection: capture length before creation
    const prevCount = ctx.diagram.nodes.length;
    ctx.diagram.addSubGraph(position.x, position.y);

    if (ctx.diagram.nodes.length <= prevCount) {
      throw new Error("Failed to create subflow: no node was added");
    }
    const created = ctx.diagram.nodes[ctx.diagram.nodes.length - 1];

    // Apply label if provided
    if (label) {
      ctx.diagram.updateModule(created.id, { name: label, label });
    }

    return {
      nodeId: created.id,
      name:
        ((created.data as Record<string, unknown>)?.name as string) ??
        ((created.data as Record<string, unknown>)?.label as string) ??
        created.id,
    };
  },
};
