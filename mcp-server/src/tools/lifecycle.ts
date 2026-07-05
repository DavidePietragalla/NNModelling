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
