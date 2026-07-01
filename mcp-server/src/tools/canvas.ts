/**
 * Canvas / Viewport Tools — informational stubs for canvas state queries.
 *
 * ⚠️ IMPORTANT: Svelte Flow viewport state (zoom, pan) is browser-only.
 * These tools return stub values because the MCP server does not own the
 * canvas viewport — it is managed by the browser's Svelte Flow instance.
 *
 * For actual canvas state, use the glimpse MCP tool (screenshot) to observe
 * the current viewport visually.
 *
 * @module tools/canvas
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Helpers ─────────────────────────────────────────────────────────────

/**
 * Stub: Canvas state is browser-only.
 * Returns a static default viewport.
 */
const STUB_CANVAS_STATE = { zoom: 1, x: 0, y: 0 } as const;

// ── Schemas ─────────────────────────────────────────────────────────────

export const get_canvas_state = {
  schema: z.object({}),

  async handler(
    _ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{ zoom: number; x: number; y: number }> {
    /**
     * STUB — Canvas state is browser-only.
     * Svelte Flow manages zoom and pan on the client side.
     * The MCP server does not have access to the browser viewport.
     * For actual viewport information, use the glimpse screenshot tool.
     */
    return { ...STUB_CANVAS_STATE };
  },
};

export const fit_view = {
  schema: z.object({
    nodeIds: z.array(z.string()).optional(),
  }),

  async handler(
    _ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{ success: boolean; note: string }> {
    /**
     * STUB — fit_view is a browser-side operation.
     * Svelte Flow's fitView() is called on the browser's FlowCanvas instance.
     * The MCP server cannot trigger this; the LLM should use glimpse to
     * instruct the browser or perform the viewport operation manually.
     */
    return {
      success: true,
      note: "fit_view is a browser-side operation — use the browser UI or instruct the user to fit the view",
    };
  },
};

export const center_view = {
  schema: z.object({
    x: z.number().optional(),
    y: z.number().optional(),
    zoom: z.number().optional(),
  }),

  async handler(
    _ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{ success: boolean; note: string }> {
    /**
     * STUB — center_view is a browser-side operation.
     * Svelte Flow's setCenter() is called on the browser's FlowCanvas instance.
     * The MCP server cannot trigger this; the LLM should use glimpse to
     * instruct the browser or perform the viewport operation manually.
     */
    return {
      success: true,
      note: "center_view is a browser-side operation — use the browser UI or instruct the user to center the view",
    };
  },
};
