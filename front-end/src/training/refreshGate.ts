/**
 * Generation guard for asynchronous training-job refreshes.
 *
 * The gate owns two independent generations:
 *
 * - **Connection epoch** — bumped by `invalidate()` whenever the connection
 *   identity changes or the owner is torn down (forget, re-activation,
 *   cleanup, client replacement). Every request begun before the bump is
 *   stale and must not touch component state.
 * - **Request sequence** — bumped by every `begin()`. Among concurrent
 *   refreshes of the same epoch only the latest one may commit its result,
 *   so an older response can never overwrite a newer one
 *   (last-request-wins).
 *
 * A handle returned by `begin()` reports `isCurrent()` only while both the
 * captured epoch and the captured sequence still match the gate. The owning
 * component applies job lists, error messages and loading flags only inside
 * `if (request.isCurrent())`, which keeps out-of-order completions and
 * post-cleanup responses from corrupting live state.
 */
export class RefreshGate {
  private epoch = 0;
  private seq = 0;

  /** Invalidate every in-flight request (forget, reconnect, cleanup). */
  invalidate(): void {
    this.epoch += 1;
  }

  /** Begin a new refresh; only the latest begun request can still commit. */
  begin(): RefreshHandle {
    const epoch = this.epoch;
    const seq = ++this.seq;
    return {
      isCurrent: () => this.epoch === epoch && this.seq === seq,
    };
  }
}

export interface RefreshHandle {
  /** True while this request is the latest of the current connection epoch. */
  isCurrent(): boolean;
}
