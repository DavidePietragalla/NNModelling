/**
 * Server Lifecycle Tools — reset diagram state and health check.
 *
 * `reset_diagram` provides a hard reset of the entire diagram state:
 * clears nodes, edges, undo/redo history, and the event bus buffer.
 * This is equivalent to starting fresh with an empty canvas.
 *
 * `ping` is a lightweight health check that returns server status
 * including uptime, node/edge count, and active transaction info.
 *
 * @module tools/lifecycle
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Schemas ────────────────────────────────────────────────────────────

export const reset_diagram = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    success: boolean;
    message: string;
  }> {
    // Clear diagram state by replacing arrays with empty ones
    ctx.diagram.nodes = [];
    ctx.diagram.edges = [];

    // Clear undo/redo history
    ctx.history.clear();

    // Clear the event bus buffer and reset sequence counter
    ctx.diagram.events.clear();

    // Reset the last event cursor so event polling starts fresh
    ctx.lastEventCursor = 0;

    return {
      success: true,
      message: "Diagram reset to empty state",
    };
  },
};

export const ping = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    status: string;
    uptime: number;
    nodeCount: number;
    edgeCount: number;
    activeTransaction: string | null;
  }> {
    return {
      status: "ok",
      uptime: process.uptime(),
      nodeCount: ctx.diagram.nodes.length,
      edgeCount: ctx.diagram.edges.length,
      activeTransaction: ctx.transactions.getActiveId(),
    };
  },
};
