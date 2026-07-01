#!/usr/bin/env node

/**
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
  const { server, ctx } = await createServer(stereotypesDir);

  // ── Connect stdio transport (MCP protocol) ─────────────────────────
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("[nnmodelling-mcp] Server connected via stdio");

  // ── Graceful shutdown ──────────────────────────────────────────────
  const shutdown = async (): Promise<void> => {
    console.error("[nnmodelling-mcp] Shutting down...");

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
