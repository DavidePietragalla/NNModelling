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
