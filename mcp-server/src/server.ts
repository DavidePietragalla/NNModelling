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
import * as pipelineMod from "./pipeline";
import { BrowserRPCClient } from "./browser-client";

// ── Import all tool modules ─────────────────────────────────────────────
// Each file exports multiple named tools (e.g. create_node, delete_nodes).
// We iterate over Object.entries to discover them automatically.
import * as graphTools from "./tools/graph";
import * as paramTools from "./tools/parameters";
import * as selectionTools from "./tools/selection";
import * as canvasTools from "./tools/canvas";
import * as validationTools from "./tools/validation";
import * as conversionTools from "./tools/conversion";
import * as inspectionTools from "./tools/inspection";
import * as lifecycleTools from "./tools/lifecycle";

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

// ── Internal Types ──────────────────────────────────────────────────────

interface MCPToolEntry {
  schema: Record<string, unknown>;
  handler: (ctx: ServerContext, input: Record<string, unknown>) => Promise<unknown>;
}

/**
 * Minimal Zod-to-JSON-Schema converter.
 *
 * Covers the subset of Zod types used by NNModelling tool schemas:
 *   ZodObject, ZodString, ZodNumber, ZodBoolean, ZodArray,
 *   ZodOptional, ZodEnum, ZodRecord
 *
 * Uses Zod's internal `_def.typeName` property, which is stable across
 * Zod v3.x minor versions. Returns a plain JSON Schema object suitable
 * for the MCP ListToolsResultSchema.
 */
function zodToJsonSchema(schema: unknown): Record<string, unknown> {
  if (!schema || typeof schema !== "object") return {};

  const def = (schema as Record<string, unknown>)._def as Record<string, unknown> | undefined;
  if (!def || typeof def !== "object") return {};

  const typeName = def.typeName as string | undefined;
  if (!typeName) return {};

  switch (typeName) {
    case "ZodObject": {
      const shape = (def as Record<string, unknown>).shape as Record<string, unknown> | undefined;
      if (!shape) return { type: "object", properties: {}, required: [] };

      const properties: Record<string, unknown> = {};
      const required: string[] = [];

      for (const [key, value] of Object.entries(shape)) {
        properties[key] = zodToJsonSchema(value);
        const innerDef = (value as Record<string, unknown>)?._def as Record<string, unknown> | undefined;
        const innerTypeName = innerDef?.typeName as string | undefined;
        if (!innerTypeName || !innerTypeName.includes("ZodOptional")) {
          required.push(key);
        }
      }

      return { type: "object", properties, required } satisfies Record<string, unknown>;
    }

    case "ZodString":
      return { type: "string" };

    case "ZodNumber":
      return { type: "number" };

    case "ZodBoolean":
      return { type: "boolean" };

    case "ZodArray": {
      const innerType = (def as Record<string, unknown>).type as unknown | undefined;
      return { type: "array", items: zodToJsonSchema(innerType) };
    }

    case "ZodOptional": {
      const innerType = (def as Record<string, unknown>).innerType as unknown | undefined;
      return zodToJsonSchema(innerType);
    }

    case "ZodEnum": {
      const values = (def as Record<string, unknown>).values as string[] | undefined;
      return { type: "string", enum: values ?? [] };
    }

    case "ZodRecord": {
      const valueType = (def as Record<string, unknown>).valueType as unknown | undefined;
      return { type: "object", additionalProperties: zodToJsonSchema(valueType) };
    }

    default:
      return { type: "string" };
  }
}

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
        schema: zodToJsonSchema(entry.schema),
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
 * @returns An object with the MCP `Server` instance and the shared `ServerContext`.
 */
export async function createServer(
  stereotypesDir: string,
): Promise<{ server: Server; ctx: ServerContext }> {
  // ── Step 1: Load stereotypes (static data) ──────────────────────────
  console.error(`[nnmodelling-mcp] Loading stereotypes from ${stereotypesDir}`);
  const stereotypes = loadStereotypesFromDirectory(stereotypesDir);
  console.error(`[nnmodelling-mcp] Loaded ${stereotypes.length} stereotypes`);

  // ── Step 2: Create BrowserRPCClient ─────────────────────────────────
  // (Browser connection will be awaited in Phase 4)
  const browser = new BrowserRPCClient();

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

  return { server, ctx };
}
