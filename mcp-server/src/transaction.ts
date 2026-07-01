/**
 * TransactionManager — buffers mutations during a transaction and applies
 * them atomically on commit, or discards them on rollback via snapshot restore.
 *
 * ## Usage
 *
 * ```typescript
 * const tx = new TransactionManager(diagram);
 *
 * const txId = tx.begin("create two nodes");
 * tx.buffer({ type: "add_node", execute: () => diagram.addModule(...) });
 * tx.buffer({ type: "add_edge", execute: () => diagram.addEdge(...) });
 *
 * // Commit: all mutations applied in order
 * const result = tx.commit();
 *
 * // Or rollback: state restored to pre-begin snapshot
 * // tx.rollback();
 * ```
 */

import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import type { DiagramCoreSnapshot } from "@nnmodelling/front-end/core/types";
import {
  NoActiveTransactionError,
  TransactionAlreadyActiveError,
} from "./errors";

// ── Interfaces ──────────────────────────────────

export interface BufferedMutation {
  /** Human-readable label describing the mutation (e.g. "add_node"). */
  type: string;
  /** Execute the mutation against the DiagramCore. Called on commit. */
  execute: () => void;
  /** Optional undo action. Not currently used (rollback uses snapshot). */
  undo?: () => void;
}

interface Transaction {
  id: string;
  label: string;
  snapshot: DiagramCoreSnapshot;
  mutations: BufferedMutation[];
}

// ── TransactionManager ──────────────────────────

export class TransactionManager {
  private active: Transaction | null = null;
  private diagram: DiagramCore;

  constructor(diagram: DiagramCore) {
    this.diagram = diagram;
  }

  /**
   * Start a new transaction.
   * Captures a pre-transaction snapshot (used by rollback to restore state).
   *
   * @param label - Human-readable description of the transaction.
   * @returns The transaction ID (UUID).
   * @throws {TransactionAlreadyActiveError} if a transaction is already active.
   */
  begin(label: string): string {
    if (this.active) {
      throw new TransactionAlreadyActiveError();
    }
    this.active = {
      id: crypto.randomUUID(),
      label,
      snapshot: this.diagram.getSnapshot(),
      mutations: [],
    };
    return this.active.id;
  }

  /**
   * Buffer a mutation for the active transaction, or execute immediately
   * if no transaction is active.
   *
   * - **Active transaction**: the mutation is queued; it will be applied
   *   when `commit()` is called.
   * - **No active transaction**: the mutation is executed immediately
   *   (fire-and-forget).
   *
   * @param mutation - The mutation to buffer or execute.
   */
  buffer(mutation: BufferedMutation): void {
    if (!this.active) {
      // No active transaction: execute immediately
      mutation.execute();
      return;
    }
    this.active.mutations.push(mutation);
  }

  /**
   * Commit the active transaction.
   * Applies all buffered mutations in FIFO order, then clears the transaction.
   *
   * @returns Summary with the transaction ID and an array of applied mutations.
   * @throws {NoActiveTransactionError} if no transaction is active.
   */
  commit(): {
    transactionId: string;
    mutations: Array<{ type: string; summary: string }>;
  } {
    if (!this.active) {
      throw new NoActiveTransactionError();
    }
    const tx = this.active;

    // Apply all buffered mutations in order
    for (const mutation of tx.mutations) {
      mutation.execute();
    }

    const summary = tx.mutations.map((m) => ({
      type: m.type,
      summary: `${m.type}`,
    }));

    this.active = null;
    return { transactionId: tx.id, mutations: summary };
  }

  /**
   * Roll back the active transaction.
   * Restores the diagram to the pre-transaction snapshot, discarding all
   * buffered mutations.
   *
   * @returns Summary with the transaction ID and the number of discarded mutations.
   * @throws {NoActiveTransactionError} if no transaction is active.
   */
  rollback(): { transactionId: string; discardedMutations: number } {
    if (!this.active) {
      throw new NoActiveTransactionError();
    }
    const tx = this.active;

    // Restore pre-transaction snapshot
    this.diagram.restoreSnapshot(tx.snapshot);

    const count = tx.mutations.length;
    this.active = null;
    return { transactionId: tx.id, discardedMutations: count };
  }

  /**
   * Check whether a transaction is currently active.
   */
  isActive(): boolean {
    return this.active !== null;
  }

  /**
   * Get the ID of the active transaction, or `null` if none is active.
   */
  getActiveId(): string | null {
    return this.active?.id ?? null;
  }
}
