/**
 * Event Polling Tool — retrieve domain events from the EventBus ring buffer.
 *
 * The MCP server maintains an internal EventBus on DiagramCore that records
 * every mutation as a typed DomainEvent with monotonically increasing sequence
 * numbers. The `get_events` tool allows the LLM to poll for new events since
 * the last known sequence number, optionally filtered by event type.
 *
 * This is the primary mechanism for the LLM to observe what mutations have
 * occurred (e.g., after a transaction commit or after the browser user edits
 * the diagram via the UI).
 *
 * @module tools/events
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import type { DomainEventType, DomainEvent } from "@nnmodelling/front-end/core/types";

// ── Schema ────────────────────────────────────────────────────────────

export const get_events = {
  schema: z.object({
    /** Sequence number to get events after (exclusive). If omitted, return all buffered events. */
    since: z.number().int().min(0).optional(),
    /** Filter to specific event types. If omitted, return all types. */
    types: z
      .array(
        z.enum([
          "node_created",
          "node_deleted",
          "node_updated",
          "node_moved",
          "edge_created",
          "edge_deleted",
          "edge_reconnected",
          "subflow_toggled",
          "selection_changed",
          "graph_changed",
          "diagram_reset",
          "diagram_imported",
        ]),
      )
      .optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<{
    events: DomainEvent[];
    latestSeq: number;
    count: number;
  }> {
    const { since, types } = input;

    // Retrieve all events since the given sequence (or from the beginning of the buffer)
    const allEvents = ctx.diagram.events.getEventsSince(since ?? 0);

    // Filter by event types if requested
    const filtered = types
      ? allEvents.filter((e) => types.includes(e.type as DomainEventType))
      : allEvents;

    // Determine the latest sequence number from the EventBus
    const latestSeq = ctx.diagram.events.getCurrentSeq();

    // Update the cursor so the LLM can track what it has already seen
    ctx.lastEventCursor = latestSeq;

    return {
      events: filtered,
      latestSeq,
      count: filtered.length,
    };
  },
};
