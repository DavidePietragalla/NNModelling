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
 * MCP Server Bootstrap — creates BrowserRPCClient, loads stereotypes,
 * registers all tools, and returns the MCP Server instance and context.
 *
 * This is the wiring hub of the NNModelling MCP server. It:
 *   1. Loads stereotypes via StereotypeCore.loadFromDirectoryNode
 *   2. Creates a BrowserRPCClient for browser communication
 *   3. Creates the MCP Server instance
 *   4. Registers all tools from tools/*.ts (iterates exports, finds {schema,handler} pairs)
 *   5. Implements ListToolsRequestSchema and CallToolRequestSchema
 *
 * The browser is the single source of truth for diagram state.
 * The server is a thin proxy — it sends RPC calls to the browser
 * and forwards results back to the LLM via MCP.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import * as pipelineMod from "./pipeline.js";
import { BrowserRPCClient } from "./browser-client.js";

// ── Import all tool modules ─────────────────────────────────────────────
// Each file exports multiple named tools (e.g. create_node, delete_nodes).
// We iterate over Object.entries to discover them automatically.
import * as graphTools from "./tools/graph.js";
import * as paramTools from "./tools/parameters.js";
import * as selectionTools from "./tools/selection.js";
import * as canvasTools from "./tools/canvas.js";
import * as validationTools from "./tools/validation.js";
import * as conversionTools from "./tools/conversion.js";
import * as inspectionTools from "./tools/inspection.js";
import * as lifecycleTools from "./tools/lifecycle.js";
import * as connectionTools from "./tools/connection.js";
import * as screenshotTools from "./tools/screenshot.js";

// ── ServerContext ───────────────────────────────────────────────────────

/**
 * Shared context object passed as the first argument to every MCP tool handler.
 * Provides access to:
 *   - browser:      BrowserRPCClient for sending RPC calls to the browser
 *   - pipeline:     Python subprocess interface (executeConversion, etc.)
 *   - stereotypes:  Static stereotype definitions loaded at startup
 */
export interface ServerContext {
  browser: BrowserRPCClient;
  pipeline: typeof pipelineMod;
  stereotypes: StereotypeCore[];
}

export interface CreateServerOptions {
  wsPort?: number;
}

// ── Internal Types ──────────────────────────────────────────────────────

interface MCPToolEntry {
  schema: Record<string, unknown>;
  handler: (ctx: ServerContext, input: Record<string, unknown>) => Promise<unknown>;
}

import { zodToJsonSchema } from "zod-to-json-schema";

/**
 * Discover tool entries from a module's named exports.
 *
 * Each tool file exports multiple named constants matching the pattern:
 * ```ts
 * export const tool_name = {
 *   schema: z.object({...}),
 *   async handler(ctx: ServerContext, input: ...) { ... }
 * };
 * ```
 *
 * This function iterates Object.entries and collects any value
 * that has both `schema` and `handler` properties.
 */
function discoverTools(module: Record<string, unknown>): Map<string, MCPToolEntry> {
  const tools = new Map<string, MCPToolEntry>();

  for (const [name, value] of Object.entries(module)) {
    if (
      value &&
      typeof value === "object" &&
      "schema" in (value as Record<string, unknown>) &&
      "handler" in (value as Record<string, unknown>)
    ) {
      const entry = value as { schema: unknown; handler: (ctx: ServerContext, input: Record<string, unknown>) => Promise<unknown> };
      tools.set(name, {
        schema: zodToJsonSchema(entry.schema as any),
        handler: entry.handler,
      });
    }
  }

  return tools;
}

// ── ESM-compatible stereotype loader ────────────────────────────────────
// StereotypeCore.loadFromDirectoryNode() uses require("fs") which fails in
// ESM. This local function uses statically-imported readdirSync/readFileSync
// to achieve the same result without CJS interop.

function loadStereotypesFromDirectory(stereotypesDir: string): StereotypeCore[] {
  const loaded: StereotypeCore[] = [];

  function walkDir(dir: string): void {
    const entries = readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        walkDir(fullPath);
      } else if (entry.name.endsWith(".json")) {
        try {
          const content = readFileSync(fullPath, "utf-8");
          const jsonData = JSON.parse(content);
          loaded.push(new StereotypeCore(fullPath, jsonData));
        } catch (e) {
          console.error(`Error loading stereotype from ${fullPath}:`, e);
        }
      }
    }
  }

  walkDir(stereotypesDir);
  return loaded.sort((a, b) => a.name.localeCompare(b.name));
}

// ── createServer ────────────────────────────────────────────────────────

/**
 * Create and initialize the NNModelling MCP server.
 *
 * @param stereotypesDir - Absolute path to the Stereotypes/ directory
 *   (containing Modules/, Joins/, SubFlows/ subdirectories).
 * @returns An object with the MCP `Server` instance, shared `ServerContext`, and `BrowserRPCClient`.
 */
export async function createServer(
  stereotypesDir: string,
  options: CreateServerOptions = {},
): Promise<{ server: Server; ctx: ServerContext; browser: BrowserRPCClient }> {
  // ── Step 1: Load stereotypes (static data) ──────────────────────────
  console.error(`[nnmodelling-mcp] Loading stereotypes from ${stereotypesDir}`);
  const stereotypes = loadStereotypesFromDirectory(stereotypesDir);
  console.error(`[nnmodelling-mcp] Loaded ${stereotypes.length} stereotypes`);

  // ── Step 2: Create BrowserRPCClient and start listening ────────────
  const browser = new BrowserRPCClient({ port: options.wsPort });
  await browser.start();
  console.error("[nnmodelling-mcp] Browser WebSocket server ready");

  // ── Step 3: Build ServerContext ──────────────────────────────────────
  const ctx: ServerContext = {
    browser,
    pipeline: pipelineMod,
    stereotypes,
  };

  // ── Step 4: Create MCP Server instance ──────────────────────────────
  const server = new Server(
    { name: "nnmodelling-mcp-server", version: "0.1.0" },
    { capabilities: { tools: {} } }, // No resources capability
  );

  // ── Step 5: Register all tools ──────────────────────────────────────
  const toolRegistry = new Map<string, MCPToolEntry>();

  // Merge tools from all modules. Duplicate names are overwritten
  // (last module wins — none should collide across files).
  const allToolModules = [
    graphTools,
    paramTools,
    selectionTools,
    canvasTools,
    validationTools,
    conversionTools,
    inspectionTools,
    lifecycleTools,
    connectionTools,
    screenshotTools,
  ] as Record<string, unknown>[];

  for (const module of allToolModules) {
    const discovered = discoverTools(module);
    for (const [name, entry] of discovered) {
      toolRegistry.set(name, entry);
    }
  }

  console.error(`[nnmodelling-mcp] Registered ${toolRegistry.size} tools`);

  // ── Step 6: ListTools handler ───────────────────────────────────────
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const tools = Array.from(toolRegistry.entries()).map(([name, tool]) => ({
      name,
      description: `NNModelling tool: ${name}`,
      inputSchema: tool.schema,
    }));

    return { tools };
  });

  // ── Step 7: CallTool handler ────────────────────────────────────────
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const tool = toolRegistry.get(request.params.name);
    if (!tool) {
      throw new Error(`Unknown tool: ${request.params.name}`);
    }

    try {
      const result = await tool.handler(ctx, (request.params.arguments ?? {}) as Record<string, unknown>);

      return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
      };
    } catch (err: unknown) {
      const typedErr = err as Record<string, unknown> | undefined;
      const error = typedErr?.code
        ? { code: typedErr.code as string, message: (typedErr.message as string) ?? "Unknown error", details: typedErr.details as Record<string, unknown> | undefined }
        : { code: "INTERNAL_ERROR", message: (err as Error)?.message ?? "Unknown error" };

      return {
        content: [{ type: "text", text: JSON.stringify({ error }) }],
        isError: true,
      };
    }
  });

  return { server, ctx, browser };
}
