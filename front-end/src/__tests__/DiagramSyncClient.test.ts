// front-end/src/__tests__/DiagramSyncClient.test.ts
// Unit tests for DiagramSyncClient — applies delta operations to $state.raw arrays.

import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { Diagram } from "../Diagram.svelte";
import { DiagramSyncClient } from "../sync/DiagramSyncClient";

// ── Mock WebSocket ─────────────────────────────────────────────────
// Must be a real class/constructor so `new WebSocket(...)` works.
class MockWebSocket {
  public close = vi.fn();
  public send = vi.fn();
  public readyState: number = WebSocket.OPEN;
  public onopen: (() => void) | null = null;
  public onclose: (() => void) | null = null;
  public onerror: ((e: Event) => void) | null = null;
  public onmessage: ((e: MessageEvent) => void) | null = null;

  constructor(_url: string) {
    // Simulate async open
    setTimeout(() => this.onopen?.(), 0);
  }

  static OPEN: number = 1;
  static CONNECTING: number = 0;
  static CLOSING: number = 2;
  static CLOSED: number = 3;
}

// ── Helpers ────────────────────────────────────────────────────────

/**
 * Read a private property from a DiagramSyncClient instance.
 */
function readPrivate<T>(client: DiagramSyncClient, key: string): T {
  return (client as unknown as Record<string, T>)[key];
}

/**
 * Set a private property on a DiagramSyncClient instance.
 */
function writePrivate<T>(client: DiagramSyncClient, key: string, value: T): void {
  (client as unknown as Record<string, T>)[key] = value;
}

/**
 * Call a private method on a DiagramSyncClient instance with the correct `this`.
 */
function callPrivate<T>(client: DiagramSyncClient, key: string, ...args: unknown[]): T {
  const fn = (client as unknown as Record<string, Function>)[key];
  return fn.call(client, ...args) as T;
}

// ── Setup ──────────────────────────────────────────────────────────

beforeEach(() => {
  globalThis.window = { innerWidth: 1024 } as unknown as Window & typeof globalThis;
});

afterEach(() => {
  delete (globalThis as Record<string, unknown>).window;
  vi.unstubAllGlobals();
});

// ── Tests ──────────────────────────────────────────────────────────

describe("DiagramSyncClient — construction", () => {
  it("creates a client and stores diagram reference", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");
    expect(client).toBeInstanceOf(DiagramSyncClient);
  });

  it("disconnect is safe when ws is null", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");
    expect(() => client.disconnect()).not.toThrow();
  });

  it("connect and disconnect cycle works with mock WebSocket", () => {
    vi.stubGlobal("WebSocket", MockWebSocket);

    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    client.connect();
    client.disconnect();

    // After disconnect, ws should be null
    expect(readPrivate<unknown>(client, "ws")).toBeNull();
  });
});

describe("DiagramSyncClient — applySnapshot", () => {
  it("replaces diagram nodes and edges", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applySnapshot", {
      type: "snapshot",
      seq: 5,
      nodes: [
        { id: "n1", type: "custom", position: { x: 0, y: 0 }, data: {} },
        { id: "n2", type: "custom", position: { x: 100, y: 0 }, data: {} },
      ],
      edges: [{ id: "e1", source: "n1", target: "n2" }],
    });

    expect(diagram.nodes).toHaveLength(2);
    expect(diagram.edges).toHaveLength(1);
    expect(readPrivate<number>(client, "lastSeenSeq")).toBe(5);
  });
});

describe("DiagramSyncClient — applyDelta", () => {
  it("applies node_added operation", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    // lastSeenSeq = 0 means fresh connect — skip gap detection
    writePrivate(client, "lastSeenSeq", 0);

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "node_added",
          nodeId: "test-1",
          data: {
            type: "custom",
            position: { x: 10, y: 20 },
            data: { name: "TestNode" },
          },
        },
      ],
    });

    expect(diagram.nodes.some((n) => n.id === "test-1")).toBe(true);
    const added = diagram.nodes.find((n) => n.id === "test-1")!;
    expect(added.position).toEqual({ x: 10, y: 20 });
  });

  it("applies node_removed operation", () => {
    const diagram = new Diagram();
    const inputNode = diagram.nodes[0];
    expect(inputNode).toBeDefined();

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [{ op: "node_removed", nodeId: inputNode.id }],
    });

    expect(diagram.nodes.some((n) => n.id === inputNode.id)).toBe(false);
  });

  it("applies node_moved operation", () => {
    const diagram = new Diagram();
    const inputNode = diagram.nodes[0];
    const originalPos = { ...inputNode.position };

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "node_moved",
          nodeId: inputNode.id,
          position: { x: 999, y: 888 },
        },
      ],
    });

    const moved = diagram.nodes.find((n) => n.id === inputNode.id)!;
    expect(moved.position).toEqual({ x: 999, y: 888 });
    expect(moved.position).not.toEqual(originalPos);
  });

  it("applies node_updated operation", () => {
    const diagram = new Diagram();
    const inputNode = diagram.nodes[0];

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "node_updated",
          nodeId: inputNode.id,
          changes: { color: "#ff0000", customParam: "test" },
        },
      ],
    });

    const updated = diagram.nodes.find((n) => n.id === inputNode.id)!;
    expect(updated.data).toBeDefined();
    expect(updated.data.color).toBe("#ff0000");
    expect(updated.data.customParam).toBe("test");
  });

  it("applies edge_added operation", () => {
    const diagram = new Diagram();
    const n1 = diagram.nodes[0];

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "edge_added",
          edgeId: "new-edge-1",
          data: { source: n1.id, target: n1.id },
        },
      ],
    });

    expect(diagram.edges.some((e) => e.id === "new-edge-1")).toBe(true);
    const added = diagram.edges.find((e) => e.id === "new-edge-1")!;
    expect(added.source).toBe(n1.id);
  });

  it("applies edge_removed operation", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    // Add an edge first
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "edge_added",
          edgeId: "remove-me",
          data: { source: "a", target: "b" },
        },
      ],
    });

    expect(diagram.edges.some((e) => e.id === "remove-me")).toBe(true);

    // Now remove it
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 2,
      operations: [{ op: "edge_removed", edgeId: "remove-me" }],
    });

    expect(diagram.edges.some((e) => e.id === "remove-me")).toBe(false);
  });

  it("applies edge_reconnected operation", () => {
    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    // Add an edge
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "edge_added",
          edgeId: "reco-edge",
          data: { source: "a", target: "b" },
        },
      ],
    });

    // Reconnect it
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 2,
      operations: [
        {
          op: "edge_reconnected",
          edgeId: "reco-edge",
          changes: { source: "c", target: "d" },
        },
      ],
    });

    const edge = diagram.edges.find((e) => e.id === "reco-edge")!;
    expect(edge.source).toBe("c");
    expect(edge.target).toBe("d");
  });

  it("applies selection_changed operation", () => {
    const diagram = new Diagram();
    const n1 = diagram.nodes[0];

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "selection_changed",
          nodeIds: [n1.id],
          edgeIds: [],
        },
      ],
    });

    expect(diagram.nodes.find((n) => n.id === n1.id)!.selected).toBe(true);
  });

  it("applies graph_reset operation", () => {
    const diagram = new Diagram();
    expect(diagram.nodes.length).toBeGreaterThan(0);
    expect(diagram.edges.length).toBe(0);

    const client = new DiagramSyncClient(diagram, "ws://test:9999");

    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 1,
      operations: [
        {
          op: "graph_reset",
          nodes: [
            { id: "x", type: "custom", position: { x: 0, y: 0 }, data: {} },
            { id: "y", type: "custom", position: { x: 100, y: 0 }, data: {} },
          ],
          edges: [{ id: "xy", source: "x", target: "y" }],
        },
      ],
    });

    expect(diagram.nodes).toHaveLength(2);
    expect(diagram.edges).toHaveLength(1);
    expect(diagram.nodes[0].id).toBe("x");
    expect(diagram.nodes[1].id).toBe("y");
  });
});

describe("DiagramSyncClient — sequence gap detection", () => {
  it("requests snapshot on sequence gap", () => {
    vi.stubGlobal("WebSocket", MockWebSocket);

    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");
    client.connect();

    // Get the mock ws instance and set it up
    const ws = readPrivate<MockWebSocket>(client, "ws")!;
    ws.readyState = WebSocket.OPEN; // Simulate connected

    // Set lastSeenSeq to 5 (simulate having seen up to seq 5)
    writePrivate<number>(client, "lastSeenSeq", 5);

    // Apply a delta with seq 10 (gap: expected 6, got 10)
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 10,
      operations: [
        {
          op: "node_added",
          nodeId: "gap-node",
          data: { type: "custom", position: { x: 0, y: 0 }, data: {} },
        },
      ],
    });

    // Should have requested a snapshot instead of applying
    expect(ws.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "request_snapshot" }),
    );

    // Node should NOT have been added (gap prevented application)
    expect(diagram.nodes.some((n) => n.id === "gap-node")).toBe(false);
  });

  it("does NOT request snapshot on first message (lastSeenSeq === 0)", () => {
    vi.stubGlobal("WebSocket", MockWebSocket);

    const diagram = new Diagram();
    const client = new DiagramSyncClient(diagram, "ws://test:9999");
    client.connect();

    const ws = readPrivate<MockWebSocket>(client, "ws")!;
    ws.readyState = WebSocket.OPEN;

    // lastSeenSeq is 0 (fresh connect), seq is 10 — no gap detection
    callPrivate<void>(client, "applyDelta", {
      type: "delta",
      seq: 10,
      operations: [
        {
          op: "node_added",
          nodeId: "fresh-node",
          data: { type: "custom", position: { x: 0, y: 0 }, data: {} },
        },
      ],
    });

    // Should NOT have requested snapshot (fresh connect skips gap check)
    expect(ws.send).not.toHaveBeenCalled();

    // Node SHOULD have been added
    expect(diagram.nodes.some((n) => n.id === "fresh-node")).toBe(true);
  });
});
