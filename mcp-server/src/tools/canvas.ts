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
 * Canvas / Viewport Tools — real browser-RPC calls.
 *
 * These tools were previously stubs that returned hardcoded values.
 * Now they delegate to the browser via ctx.browser.call(), where the
 * BrowserRPCHandler executes real SvelteFlow viewport operations.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Schemas ─────────────────────────────────────────────────────────────

export const get_canvas_state = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return ctx.browser.call("get_canvas_state", {});
  },
};

export const fit_view = {
  schema: z.object({ nodeIds: z.array(z.string()).optional() }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("fit_view", input);
  },
};

export const center_view = {
  schema: z.object({
    x: z.number().optional(),
    y: z.number().optional(),
    zoom: z.number().optional(),
  }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    return ctx.browser.call("center_view", input);
  },
};
