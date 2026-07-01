/**
 * Graph Validation Tools — thin browser-RPC proxies.
 *
 * Every handler delegates to the browser via ctx.browser.call().
 * The browser's DiagramCore runs all validation logic.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── validate_graph ─────────────────────────────────────────────────────

export const validate_graph = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("validate_graph", {});
  },
};

// ── validate_connections ────────────────────────────────────────────────

export const validate_connections = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("validate_connections", {});
  },
};

// ── validate_parameters ─────────────────────────────────────────────────

export const validate_parameters = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("validate_parameters", {});
  },
};

// ── validate_subflows ───────────────────────────────────────────────────

export const validate_subflows = {
  schema: z.object({
    parentId: z.string().optional(),
    maxDepth: z.number().int().positive().default(10),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("validate_subflows", input);
  },
};
