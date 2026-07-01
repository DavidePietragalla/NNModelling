/**
 * Server Lifecycle Tools — thin browser-RPC proxies.
 *
 * These tools delegate to the browser via ctx.browser.call().
 * The browser's DiagramCore owns diagram state (reset) and status (ping).
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Schemas ────────────────────────────────────────────────────────────

export const reset_diagram = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("reset_diagram", {});
  },
};

export const ping = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("ping", {});
  },
};
