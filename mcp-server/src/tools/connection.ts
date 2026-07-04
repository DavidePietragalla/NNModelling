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
