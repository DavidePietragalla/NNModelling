#!/usr/bin/env node
/**
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
 *
 * NNModelling MCP Server — Entry Point
 *
 * Bootstraps the MCP server (stdio transport). The MCP server handles
 * tool/resource requests from LLM agents and communicates with the browser
 * via WebSocket RPC through the BrowserRPCClient.
 *
 * Usage:
 *   node dist/index.js
 *
 * Environment variables:
 *   NNM_WS_PORT    — WebSocket server port (default: 9339)
 *   NNM_STEREOTYPES — Override path to Stereotypes directory
 */

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { resolve } from "path";
import { createServer } from "./server";

async function main(): Promise<void> {
  // Resolve the Stereotypes directory relative to this source file
  // (import.meta.dirname → mcp-server/src/ → up 2 levels → repo root)
  const stereotypesDir =
    process.env.NNM_STEREOTYPES ?? resolve(import.meta.dirname, "..", "..", "Stereotypes");

  console.error("[nnmodelling-mcp] Starting server...");
  console.error(`[nnmodelling-mcp] Stereotypes dir: ${stereotypesDir}`);

  // ── Create the MCP server with full tool/resource registration ──────
  const { server, ctx, browser } = await createServer(stereotypesDir);

  // ── Connect stdio transport (MCP protocol) ─────────────────────────
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("[nnmodelling-mcp] Server connected via stdio");

  // ── Graceful shutdown ──────────────────────────────────────────────
  const shutdown = async (): Promise<void> => {
    console.error("[nnmodelling-mcp] Shutting down...");

    // Close browser WebSocket connection
    browser.close();

    // Close MCP server (returns Promise<void>)
    try {
      await server.close();
    } catch (err) {
      console.error("[nnmodelling-mcp] MCP server close error:", err);
    }

    process.exit(0);
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

main().catch((err: unknown) => {
  console.error("[nnmodelling-mcp] Fatal error:", err);
  process.exit(1);
});
