/**
 * MCP Server Bootstrap — creates DiagramCore, loads stereotypes, registers
 * all tools and resources, and returns the MCP Server instance and context.
 *
 * This is the wiring hub of the NNModelling MCP server. It:
 *   1. Creates a DiagramCore instance (headless state authority)
 *   2. Loads stereotypes via StereotypeCore.loadFromDirectoryNode
 *   3. Initializes TransactionManager and HistoryManager
 *   4. Sets up an EventBus subscriber for the MCP event buffer
 *   5. Creates the MCP Server instance
 *   6. Registers all tools from tools/*.ts (iterates exports, finds {schema,handler} pairs)
 *   7. Implements ListToolsRequestSchema and CallToolRequestSchema
 *   8. Registers resources via defineResources(ctx)
 *   9. Implements ListResourcesRequestSchema and ReadResourceRequestSchema
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import { TransactionManager } from "./transaction";
import { HistoryManager } from "./history";
import * as pipelineMod from "./pipeline";
import { defineResources } from "./resources/index";
import type { DomainEvent } from "@nnmodelling/front-end/core/types";

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
import * as txnTools from "./tools/transaction";
import * as historyTools from "./tools/history";
import * as eventTools from "./tools/events";
import * as lifecycleTools from "./tools/lifecycle";

// ── ServerContext ───────────────────────────────────────────────────────

/**
 * Shared context object passed as the first argument to every MCP tool handler.
 * Provides access to:
 *   - diagram:       The live DiagramCore instance (state authority)
 *   - transactions:  TransactionManager for atomic batch operations
 *   - history:       HistoryManager for undo/redo
 *   - pipeline:      Python subprocess interface (executeConversion, etc.)
 *   - eventBuffer:   Ring buffer of emitted DomainEvents
 *   - lastEventCursor: Last seq the LLM has seen (for get_events)
 */
export interface ServerContext {
  diagram: DiagramCore;
  transactions: TransactionManager;
  history: HistoryManager;
  pipeline: typeof pipelineMod;
  eventBuffer: DomainEvent[];
  lastEventCursor: number;
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
  // ── Step 1: Initialize DiagramCore (pure TS, no Svelte) ─────────────
  console.error(`[nnmodelling-mcp] Loading stereotypes from ${stereotypesDir}`);
  const diagram = new DiagramCore();
  const stereotypes = loadStereotypesFromDirectory(stereotypesDir);
  diagram.initStereotypes(stereotypes);

  // Initialize plain arrays for headless Node.js operation
  diagram.nodes = [];
  diagram.edges = [];

  console.error(`[nnmodelling-mcp] Loaded ${stereotypes.length} stereotypes`);

  // ── Step 2: Event buffer for MCP get_events tool ────────────────────
  const eventBuffer: DomainEvent[] = [];
  diagram.events.onAny((event: DomainEvent) => {
    eventBuffer.push(event);
    if (eventBuffer.length > 1000) eventBuffer.shift();
  });

  // ── Step 3: Create managers ─────────────────────────────────────────
  const transactions = new TransactionManager(diagram);
  const history = new HistoryManager();

  // ── Step 4: Build ServerContext ──────────────────────────────────────
  const ctx: ServerContext = {
    diagram,
    transactions,
    history,
    pipeline: pipelineMod,
    eventBuffer,
    lastEventCursor: 0,
  };

  // ── Step 5: Create MCP Server instance ──────────────────────────────
  const server = new Server(
    { name: "nnmodelling-mcp-server", version: "0.1.0" },
    { capabilities: { tools: {}, resources: {} } },
  );

  // ── Step 6: Register all tools ──────────────────────────────────────
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
    txnTools,
    historyTools,
    eventTools,
    lifecycleTools,
  ] as Record<string, unknown>[];

  for (const module of allToolModules) {
    const discovered = discoverTools(module);
    for (const [name, entry] of discovered) {
      toolRegistry.set(name, entry);
    }
  }

  console.error(`[nnmodelling-mcp] Registered ${toolRegistry.size} tools`);

  // ── Step 7: ListTools handler ───────────────────────────────────────
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const tools = Array.from(toolRegistry.entries()).map(([name, tool]) => ({
      name,
      description: `NNModelling tool: ${name}`,
      inputSchema: tool.schema,
    }));

    return { tools };
  });

  // ── Step 8: CallTool handler ────────────────────────────────────────
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

  // ── Step 9: Define resources ────────────────────────────────────────
  const resources = defineResources(ctx);

  // ── Step 10: ListResources handler ───────────────────────────────────
  server.setRequestHandler(ListResourcesRequestSchema, async () => ({
    resources: resources.map((r) => ({
      uri: r.uri,
      name: r.name,
      description: r.description,
      mimeType: r.mimeType,
    })),
  }));

  // ── Step 11: ReadResource handler ────────────────────────────────────
  server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const uri = new URL(request.params.uri);

    for (const resource of resources) {
      // Build a regex from the URI template (replace {param} with wildcard capture)
      const pattern = resource.uri.replace(/\{[^}]+\}/g, "[^/]+");
      const regex = new RegExp(`^${pattern}$`);

      if (regex.test(request.params.uri)) {
        return await resource.read(uri);
      }
    }

    throw new Error(`Resource not found: ${request.params.uri}`);
  });

  return { server, ctx };
}
