/**
 * Selection Tools — thin browser-RPC proxies.
 *
 * Every handler delegates to the browser via ctx.browser.call().
 * Selection state is managed by the browser's DiagramCore.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Schemas ────────────────────────────────────────────────────────────

export const select_nodes = {
  schema: z.object({
    nodeIds: z.array(z.string()),
    mode: z.enum(["replace", "add", "remove"]).optional().default("replace"),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("select_nodes", input);
  },
};

export const clear_selection = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("clear_selection", {});
  },
};

export const get_selection = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_selection", {});
  },
};

export const select_all = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("select_all", {});
  },
};
