/**
 * Transaction Control Tools — begin, commit, and rollback atomic batches
 * of diagram mutations.
 *
 * Transactions are managed by the `TransactionManager` on the `ServerContext`.
 * During an active transaction, mutations are buffered rather than applied
 * immediately. On `commit`, all buffered mutations are applied atomically
 * and a history snapshot is pushed. On `rollback`, the diagram state is
 * restored to the pre-transaction snapshot.
 *
 * ## Usage
 *
 * 1. Call `begin_transaction` with a descriptive label.
 * 2. Perform mutations (create_node, connect_nodes, set_parameter, etc.).
 * 3. Call `commit` to apply all mutations and make them undoable, or
 *    `rollback` to discard them and restore the pre-transaction state.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import {
  NoActiveTransactionError,
  TransactionAlreadyActiveError,
} from "../errors";

// ── Tools ──────────────────────────────────────────────────────────────

export const begin_transaction = {
  schema: z.object({
    label: z.string().min(1, "Transaction label is required"),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ transactionId: string }> {
    const transactionId = ctx.transactions.begin(input.label);
    return { transactionId };
  },
};

export const commit = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    transactionId: string;
    mutations: Array<{ type: string; summary: string }>;
  }> {
    // Commit the transaction — applies all buffered mutations in FIFO order
    const result = ctx.transactions.commit();

    // Push a history snapshot so the entire transaction is undoable as one step
    ctx.history.pushSnapshot(
      `commit transaction: ${result.transactionId}`,
      ctx.diagram
    );

    return {
      transactionId: result.transactionId,
      mutations: result.mutations,
    };
  },
};

export const rollback = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>
  ): Promise<{
    transactionId: string;
    discardedMutations: number;
  }> {
    const result = ctx.transactions.rollback();

    return {
      transactionId: result.transactionId,
      discardedMutations: result.discardedMutations,
    };
  },
};
