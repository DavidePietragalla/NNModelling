#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { resolve } from "path";
import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { createWSServer } from "./ws-server";
import type { ServerContext } from "./server";
import type { DomainEvent } from "@nnmodelling/front-end/core/types";
import { TransactionManager } from "./transaction";
import { HistoryManager } from "./history";
import * as pipelineMod from "./pipeline";

async function main() {
  // Resolve the Stereotypes directory relative to the project root
  // In development: mcp-server/../Stereotypes/
  // In production: depends on install location
  const stereotypesDir = resolve(process.cwd(), "..", "Stereotypes");

  console.error("[nnmodelling-mcp] Starting server...");
  console.error(`[nnmodelling-mcp] Stereotypes dir: ${stereotypesDir}`);

  // ── Initialize DiagramCore ──────────────────────────────────
  const diagram = new DiagramCore();
  const stereotypes = StereotypeCore.loadFromDirectoryNode(stereotypesDir);
  diagram.initStereotypes(stereotypes);

  // Initialize nodes/edges arrays (Diagram.svelte.ts normally does this
  // with $state.raw; here we use plain arrays for the Node.js context).
  diagram.nodes = [];
  diagram.edges = [];

  // ── Event buffer for MCP get_events tool ────────────────────
  const eventBuffer: DomainEvent[] = [];
  diagram.events.onAny((event: DomainEvent) => {
    eventBuffer.push(event);
    if (eventBuffer.length > 1000) eventBuffer.shift();
  });

  // ── Server context (shared by all MCP tool handlers) ────────
  const ctx: ServerContext = {
    diagram,
    transactions: new TransactionManager(diagram),
    history: new HistoryManager(),
    pipeline: pipelineMod,
    eventBuffer,
    lastEventCursor: 0,
  };

  // ── MCP Server (stdio) ──────────────────────────────────────
  const server = new Server(
    { name: "nnmodelling-mcp-server", version: "0.1.0" },
    { capabilities: { tools: {}, resources: {} } },
  );

  // Stub ListTools handler — will be populated in Step 6/8
  server.setRequestHandler(
    { method: "tools/list" } as any,
    async () => ({ tools: [] }),
  );

  // Stub CallTool handler
  server.setRequestHandler(
    { method: "tools/call" } as any,
    async () => ({
      content: [{ type: "text", text: JSON.stringify({ error: "Not yet implemented" }) }],
      isError: true,
    }),
  );

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("[nnmodelling-mcp] Server connected via stdio");

  // ── WebSocket Server (browser sync) ─────────────────────────
  const wsPort = parseInt(process.env.NNM_WS_PORT || "9339", 10);
  const wss = createWSServer(diagram, diagram.events, { port: wsPort });

  // ── Graceful shutdown ───────────────────────────────────────
  const shutdown = () => {
    console.error("[nnmodelling-mcp] Shutting down...");
    if ((wss as any)._shutdown) {
      (wss as any)._shutdown();
    }
    server.close();
    process.exit(0);
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

main().catch((err) => {
  console.error("[nnmodelling-mcp] Fatal error:", err);
  process.exit(1);
});
