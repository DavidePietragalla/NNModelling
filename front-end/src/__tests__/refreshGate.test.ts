import { describe, expect, it } from "vitest";
import { RefreshGate } from "../training/refreshGate";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

/** Promise whose settlement is controlled by the test. */
function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * The component commits a refresh result only when its handle is still
 * current, mirroring `refreshJobs`' `if (request.isCurrent())` guards.
 * These tests drive the gate with deferred promises so the resolution order
 * is fully deterministic.
 */
describe("RefreshGate stale-refresh protection", () => {
  it("keeps the latest begun request current and drops older ones of the same session", () => {
    const gate = new RefreshGate();
    const older = gate.begin();
    const latest = gate.begin();

    expect(latest.isCurrent()).toBe(true);
    expect(older.isCurrent()).toBe(false);
  });

  it("drops an older request that resolves after a newer one", async () => {
    const gate = new RefreshGate();
    const transitions: string[] = [];
    const olderResponse = deferred<string>();
    const newerResponse = deferred<string>();

    const older = gate.begin();
    const newer = gate.begin();
    const olderDone = olderResponse.promise.then((jobs) => {
      if (older.isCurrent()) transitions.push(`jobs:${jobs}`, "loading:false");
    });
    const newerDone = newerResponse.promise.then((jobs) => {
      if (newer.isCurrent()) transitions.push(`jobs:${jobs}`, "loading:false");
    });

    newerResponse.resolve("newer-jobs");
    await newerDone;
    olderResponse.resolve("older-jobs");
    await olderDone;

    expect(older.isCurrent()).toBe(false);
    expect(transitions).toEqual(["jobs:newer-jobs", "loading:false"]);
  });

  it("drops a refresh from connection A that resolves after switching to B", async () => {
    const gate = new RefreshGate();
    const applied: string[] = [];

    const jobsFromA = deferred<string>();
    const jobsFromB = deferred<string>();

    const refreshA = gate.begin();
    gate.invalidate(); // forget() / reconnect() bumps the epoch
    const refreshB = gate.begin();

    const appliedA = jobsFromA.promise.then((jobs) => {
      if (refreshA.isCurrent()) applied.push(jobs);
    });
    const appliedB = jobsFromB.promise.then((jobs) => {
      if (refreshB.isCurrent()) applied.push(jobs);
    });

    jobsFromA.resolve("jobs-owned-by-A");
    await appliedA;
    jobsFromB.resolve("jobs-owned-by-B");
    await appliedB;

    expect(refreshA.isCurrent()).toBe(false);
    expect(refreshB.isCurrent()).toBe(true);
    expect(applied).toEqual(["jobs-owned-by-B"]);
  });

  it("keeps a stale rejection from overwriting the current error and loading state", async () => {
    const gate = new RefreshGate();
    const transitions: string[] = [];
    const olderFailure = deferred<unknown>();
    const newerFailure = deferred<unknown>();

    const older = gate.begin();
    const newer = gate.begin();
    const olderDone = olderFailure.promise.catch((reason) => {
      if (older.isCurrent()) transitions.push(`error:${String(reason)}`, "loading:false");
    });
    const newerDone = newerFailure.promise.catch((reason) => {
      if (newer.isCurrent()) transitions.push(`error:${String(reason)}`, "loading:false");
    });

    // The newest request fails first: its error and loading reset apply.
    newerFailure.reject("session_expired");
    await newerDone;
    // The older request rejects later: it must not touch error/loading again.
    olderFailure.reject("network failure");
    await olderDone;

    expect(transitions).toEqual(["error:session_expired", "loading:false"]);
  });

  it("invalidates every in-flight request on cleanup", async () => {
    const gate = new RefreshGate();
    const applied: string[] = [];
    const response = deferred<string>();

    const request = gate.begin();
    const done = response.promise.then((jobs) => {
      if (request.isCurrent()) applied.push(jobs);
    });

    gate.invalidate(); // component cleanup / forget
    response.resolve("jobs");
    await done;

    expect(request.isCurrent()).toBe(false);
    expect(applied).toEqual([]);
  });

  it("accepts a fresh request begun after invalidation", () => {
    const gate = new RefreshGate();
    gate.invalidate();

    const request = gate.begin();

    expect(request.isCurrent()).toBe(true);
  });
});
