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
 * Multi-Tab Connection Management Tools
 *
 * These tools let the LLM discover connected browser tabs and
 * select which one subsequent RPC calls go to.
 *
 * The active tab is set automatically for the first tab that
 * connects. When additional tabs connect, the model must
 * explicitly select one.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Tools ──────────────────────────────────────────────────────────────

export const list_browser_tabs = {
  schema: z.object({}),

  async handler(ctx: ServerContext, _input: z.infer<typeof this.schema>) {
    return {
      tabs: ctx.browser.getTabs(),
      activeTabId: ctx.browser.getActiveTabId(),
    };
  },
};

export const select_browser_tab = {
  schema: z.object({ tabId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: z.infer<typeof this.schema>) {
    const tabs = ctx.browser.getTabs();
    if (!tabs.find((t) => t.id === input.tabId)) {
      throw new Error(
        `Tab '${input.tabId}' not found. Available tabs: ${tabs.map((t) => t.id).join(", ") || "none"}`,
      );
    }

    ctx.browser.selectTab(input.tabId);
    return { success: true, selectedTab: input.tabId };
  },
};
