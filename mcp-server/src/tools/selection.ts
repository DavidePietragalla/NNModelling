/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

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
