/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
        view: s.view,
      })),
    };
  },
};
