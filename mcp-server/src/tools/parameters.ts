/**
 * Parameter Manipulation Tools — thin browser-RPC proxies.
 *
 * Every handler delegates to the browser via ctx.browser.call().
 * The browser's DiagramCore validates parameters against stereotype defs.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Schemas ────────────────────────────────────────────────────────────

export const set_parameter = {
  schema: z.object({
    nodeId: z.string().min(1),
    key: z.string().min(1),
    value: z.string(),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("set_parameter", input);
  },
};

export const update_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
    params: z.record(z.string(), z.string()),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("update_parameters", input);
  },
};

export const reset_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
    keys: z.array(z.string().min(1)).optional(),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("reset_parameters", input);
  },
};

export const query_parameters = {
  schema: z.object({
    nodeId: z.union([z.string().min(1), z.array(z.string().min(1)).min(1)]),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("query_parameters", input);
  },
};
