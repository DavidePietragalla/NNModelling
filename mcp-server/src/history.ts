/**
 * Snapshot-based undo/redo system for the NNModelling MCP server.
 *
 * Maintains two stacks (undo/redo) of DiagramSnapshot objects.
 * - `pushSnapshot` captures the current diagram state and pushes it onto the undo stack.
 * - `undo` restores the most recent snapshot from the undo stack and saves the
 *   current state onto the redo stack.
 * - `redo` restores the most recent snapshot from the redo stack and saves the
 *   current state onto the undo stack.
 *
 * The undo stack is capped at 50 entries. Pushing a new snapshot invalidates
 * (clears) the redo stack. Estimated memory: ~5MB at capacity for a typical
 * diagram of ~100 nodes.
 */

import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import type { DiagramSnapshot } from "@nnmodelling/front-end/core/types";
import { NothingToUndoError, NothingToRedoError } from "./errors";

export class HistoryManager {
  private undoStack: DiagramSnapshot[] = [];
  private redoStack: DiagramSnapshot[] = [];
  private readonly maxDepth: number = 50;

  /**
   * Capture the current diagram state and push it onto the undo stack.
   * Invalidates the redo stack (new action = branch point).
   * Drops the oldest entry if the undo stack exceeds maxDepth.
   *
   * @param description - Human-readable label for the snapshot
   *                      (e.g. "delete nodes [lin1, relu]" or "move node conv1").
   * @param diagram - The DiagramCore instance to snapshot.
   */
  pushSnapshot(description: string, diagram: DiagramCore): void {
    const snapshot = diagram.getSnapshot();
    this.undoStack.push({
      ...snapshot,
      timestamp: Date.now(),
      description,
    });

    // New action invalidates redo (branching undo)
    this.redoStack = [];

    // Cap at maxDepth — drop oldest (front of array)
    if (this.undoStack.length > this.maxDepth) {
      this.undoStack.shift();
    }
  }

  /**
   * Undo the most recent snapshot: pop from undo stack, save current state
   * to redo stack, restore the popped snapshot.
   *
   * @param diagram - The DiagramCore instance to mutate.
   * @returns Object describing what was undone and available action flags.
   * @throws {NothingToUndoError} If the undo stack is empty.
   */
  undo(
    diagram: DiagramCore
  ): { undone: string; canUndo: boolean; canRedo: boolean } {
    if (this.undoStack.length === 0) throw new NothingToUndoError();

    // Save current state to redo stack before restoring
    const current = diagram.getSnapshot();
    this.redoStack.push({
      ...current,
      timestamp: Date.now(),
      description: "undo point",
    });

    // Pop and restore the previous snapshot
    const snapshot = this.undoStack.pop()!;
    diagram.restoreSnapshot(snapshot);

    return {
      undone: snapshot.description,
      canUndo: this.undoStack.length > 0,
      canRedo: this.redoStack.length > 0,
    };
  }

  /**
   * Redo a previously undone snapshot: pop from redo stack, save current
   * state to undo stack, restore the popped snapshot.
   *
   * @param diagram - The DiagramCore instance to mutate.
   * @returns Object describing what was redone and available action flags.
   * @throws {NothingToRedoError} If the redo stack is empty.
   */
  redo(
    diagram: DiagramCore
  ): { redone: string; canUndo: boolean; canRedo: boolean } {
    if (this.redoStack.length === 0) throw new NothingToRedoError();

    // Save current state to undo stack before restoring
    const current = diagram.getSnapshot();
    this.undoStack.push({
      ...current,
      timestamp: Date.now(),
      description: "redo point",
    });

    // Pop and restore the redone snapshot
    const snapshot = this.redoStack.pop()!;
    diagram.restoreSnapshot(snapshot);

    return {
      redone: snapshot.description,
      canUndo: this.undoStack.length > 0,
      canRedo: this.redoStack.length > 0,
    };
  }

  /**
   * Return the current status of both stacks, including undo stack metadata.
   */
  getStatus(): {
    undoCount: number;
    redoCount: number;
    maxUndoDepth: number;
    undoStack: Array<{
      description: string;
      timestamp: number;
      nodeCount: number;
      edgeCount: number;
    }>;
  } {
    return {
      undoCount: this.undoStack.length,
      redoCount: this.redoStack.length,
      maxUndoDepth: this.maxDepth,
      undoStack: this.undoStack.map((s) => ({
        description: s.description,
        timestamp: s.timestamp,
        nodeCount: s.nodes.length,
        edgeCount: s.edges.length,
      })),
    };
  }

  /**
   * Reset both undo and redo stacks. All history is lost.
   */
  clear(): void {
    this.undoStack = [];
    this.redoStack = [];
  }
}
