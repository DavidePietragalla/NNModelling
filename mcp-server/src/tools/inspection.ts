/**
 * Diagram Inspection Tools — thin browser-RPC proxies.
 *
 * Most tools delegate to the browser via ctx.browser.call().
 * `list_stereotypes` can fall back to the server-side cache if the
 * browser is not connected.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Tools ──────────────────────────────────────────────────────────────

export const get_graph = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_graph", {});
  },
};

export const get_node = {
  schema: z.object({ nodeId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_node", input);
  },
};

export const get_edges = {
  schema: z.object({ nodeId: z.string().optional() }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_edges", input);
  },
};

export const get_subflow = {
  schema: z.object({ parentId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_subflow", input);
  },
};

export const graph_statistics = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("graph_statistics", {});
  },
};

export const list_stereotypes = {
  schema: z.object({ category: z.string().optional() }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    // Try browser first; fall back to server-side cache
    if (ctx.browser.isConnected()) {
      try {
        return await ctx.browser.call("list_stereotypes", input);
      } catch {
        // Fall through to server-side cache
      }
    }

    const filtered = input.category
      ? ctx.stereotypes.filter((s) => s.category === input.category)
      : ctx.stereotypes;

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
