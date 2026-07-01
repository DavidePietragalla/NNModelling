/**
 * Graph Manipulation Tools — create, delete, connect, disconnect, move,
 * duplicate nodes and create subflows.
 *
 * Every mutating tool pushes a history snapshot before making changes.
 * Inputs are validated with Zod schemas; errors use the MCPServerError hierarchy.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { Node, Edge } from "@nnmodelling/front-end/core/types";
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
  ): Promise<{ deletedCount: number; reparentedNodes: number }> {
    const { nodeIds } = input;

    // Validate all nodes exist
    for (const id of nodeIds) {
      if (!ctx.diagram.getNodeById(id)) throw new NodeNotFoundError(id);
    }

    ctx.history.pushSnapshot(`delete ${nodeIds.length} nodes`, ctx.diagram);
    ctx.diagram.deleteNodes(nodeIds);

    return {
      deletedCount: nodeIds.length,
      reparentedNodes: 0,
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
    edgeId: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ removed: boolean }> {
    const { edgeId } = input;

    // Find the edge first
    const edge = ctx.diagram.edges.find((e) => e.id === edgeId);
    if (!edge) throw new EdgeNotFoundError(edgeId);

    ctx.history.pushSnapshot(`disconnect edge ${edgeId}`, ctx.diagram);
    ctx.diagram.deleteEdge(edgeId);

    return { removed: true };
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
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ duplicated: Array<{ originalId: string; newId: string }> }> {
    const { nodeIds } = input;

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

    // Track original → new ID mapping
    const idMap = new Map<string, string>();
    const duplicated: Array<{ originalId: string; newId: string }> = [];
    const OFFSET = 50;
    const prevCount = ctx.diagram.nodes.length;

    // Create copies of each original node
    for (const node of originals) {
      const stereoName = (node.data as Record<string, unknown>)?.stereotype as string | undefined;
      if (!stereoName) continue;

      const stereo = ctx.diagram.getStereotype(stereoName);
      if (!stereo) continue;

      const newX = node.position.x + OFFSET;
      const newY = node.position.y + OFFSET;

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
    }

    // Build the ID map by pairing newly created nodes with originals
    const newNodes = ctx.diagram.nodes.slice(prevCount);
    const validOriginals = originals.filter((n) => {
      const s = (n.data as Record<string, unknown>)?.stereotype as string | undefined;
      return s !== undefined && ctx.diagram.getStereotype(s);
    });

    for (let i = 0; i < Math.min(validOriginals.length, newNodes.length); i++) {
      idMap.set(validOriginals[i].id, newNodes[i].id);
      duplicated.push({ originalId: validOriginals[i].id, newId: newNodes[i].id });
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
    stereotype: z.string().optional(),
    config: z
      .object({
        name: z.string().optional(),
        color: z.string().optional(),
        params: z.record(z.string(), z.string()).optional(),
      })
      .optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ nodeId: string; name: string }> {
    const { position, config } = input;

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
    ctx.diagram.addSubGraph(position.x, position.y);

    const created = ctx.diagram.nodes[ctx.diagram.nodes.length - 1];

    // Apply optional config after creation
    if (config) {
      ctx.diagram.updateModule(created.id, {
        name: config.name,
        label: config.name,
        params: config.params,
      });
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
