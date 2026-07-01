/**
 * Undo/Redo History Tools — traverse the snapshot-based undo/redo stack.
 *
 * The HistoryManager maintains two stacks (undo/redo) of DiagramSnapshot
 * objects. Every mutating tool pushes a snapshot before making changes.
 * The `undo` tool pops from the undo stack and restores the captured state;
 * `redo` pops from the redo stack.
 *
 * Events are emitted via `diagram.events` as a side-effect of the
 * `restoreSnapshot` call within the HistoryManager — the tool handlers
 * do not need to emit events explicitly.
 */

import { z } from "zod";
import type { ServerContext } from "../server";

// ── Tools ──────────────────────────────────────────────────────────────

export const undo = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    undone: string;
    canUndo: boolean;
    canRedo: boolean;
  }> {
    // `history.undo()` calls `diagram.restoreSnapshot()` internally, which
    // emits `diagram_reset` and `graph_changed` events on the EventBus.
    const result = ctx.history.undo(ctx.diagram);

    return {
      undone: result.undone,
      canUndo: result.canUndo,
      canRedo: result.canRedo,
    };
  },
};

export const redo = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    redone: string;
    canUndo: boolean;
    canRedo: boolean;
  }> {
    // `history.redo()` calls `diagram.restoreSnapshot()` internally, which
    // emits `diagram_reset` and `graph_changed` events on the EventBus.
    const result = ctx.history.redo(ctx.diagram);

    return {
      redone: result.redone,
      canUndo: result.canUndo,
      canRedo: result.canRedo,
    };
  },
};

export const get_history_status = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    undoCount: number;
    redoCount: number;
    maxUndoDepth: number;
    undoStack: Array<{
      description: string;
      timestamp: number;
      nodeCount: number;
      edgeCount: number;
    }>;
  }> {
    return ctx.history.getStatus();
  },
};
