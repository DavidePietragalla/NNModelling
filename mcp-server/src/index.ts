#!/usr/bin/env node

/**
 * NNModelling MCP Server — Entry Point
 *
 * Bootstraps the MCP server (stdio transport) and the WebSocket server
 * (browser sync) in the same process. The MCP server handles tool/resource
 * requests from LLM agents; the WebSocket server broadcasts delta updates
 * to connected browser canvases.
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
import { createWSServer } from "./ws-server";

async function main(): Promise<void> {
  // Resolve the Stereotypes directory relative to the project root.
  // In development: mcp-server/../Stereotypes/
  // In production: adjust via NNM_STEREOTYPES env var
  const stereotypesDir =
    process.env.NNM_STEREOTYPES ?? resolve(process.cwd(), "..", "Stereotypes");

  console.error("[nnmodelling-mcp] Starting server...");
  console.error(`[nnmodelling-mcp] Stereotypes dir: ${stereotypesDir}`);

  // ── Create the MCP server with full tool/resource registration ──────
  const { server, ctx } = await createServer(stereotypesDir);

  // ── Connect stdio transport (MCP protocol) ─────────────────────────
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("[nnmodelling-mcp] Server connected via stdio");

  // ── Start WebSocket server for browser sync ────────────────────────
  const wsPort = parseInt(process.env.NNM_WS_PORT || "9339", 10);
  const wss = createWSServer(ctx.diagram, ctx.diagram.events, { port: wsPort });

  // ── Graceful shutdown ──────────────────────────────────────────────
  const shutdown = (): void => {
    console.error("[nnmodelling-mcp] Shutting down...");

    // Close WebSocket server (calls unsubscribe + clears ping timer)
    const wsShutdown = (wss as unknown as Record<string, unknown>)._shutdown;
    if (typeof wsShutdown === "function") {
      (wsShutdown as () => void)();
    }

    // Close MCP server
    server.close();

    process.exit(0);
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

main().catch((err: unknown) => {
  console.error("[nnmodelling-mcp] Fatal error:", err);
  process.exit(1);
});
