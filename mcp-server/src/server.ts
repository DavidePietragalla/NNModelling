/**
 * MCP Server Context — shared type for all tool handlers.
 *
 * This is a minimal stub exporting only the ServerContext interface.
 * The full MCP server bootstrap (createServer function, tool registration,
 * resource registration) will be implemented in Step 8.
 */

import type { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import type { DomainEvent } from "@nnmodelling/front-end/core/types";
import type { TransactionManager } from "./transaction";
import type { HistoryManager } from "./history";

/**
 * ServerContext is passed as the first argument to every MCP tool handler.
 * It provides access to:
 *   - diagram:  The live DiagramCore instance (state authority)
 *   - transactions:  TransactionManager for atomic batch operations
 *   - history:  HistoryManager for undo/redo
 *   - pipeline: Python subprocess interface (executeConversion, etc.)
 *   - eventBuffer: Ring buffer of emitted DomainEvents
 *   - lastEventCursor: Last seq the LLM has seen (for get_events)
 */
export interface ServerContext {
  diagram: DiagramCore;
  transactions: TransactionManager;
  history: HistoryManager;
  pipeline: typeof import("./pipeline");
  eventBuffer: DomainEvent[];
  lastEventCursor: number;
}
