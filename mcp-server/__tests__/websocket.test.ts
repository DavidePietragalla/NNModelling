/**
 * WebSocket Server Integration Tests
 *
 * Tests the WebSocket delta broadcaster using a real DiagramCore and
 * the ws package for both server and client. Each test suite uses a
 * dynamic port to avoid conflicts.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import path from "path";
import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { createWSServer } from "../src/ws-server";
import WebSocket from "ws";
import type { WSSnapshotMessage, WSDeltaMessage } from "@nnmodelling/front-end/core/types";

// ── Helpers ───────────────────────────────────────────────────────────────

function createDiagram(): DiagramCore {
  const diagram = new DiagramCore();
  diagram.nodes = [];
  diagram.edges = [];
  const stereotypesDir = path.resolve(__dirname, "../../Stereotypes");
  diagram.initStereotypes(StereotypeCore.loadFromDirectoryNode(stereotypesDir));
  return diagram;
}

function findAvailablePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = require("net").createServer();
    server.listen(0, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on("error", reject);
  });
}

function waitForMessage(ws: WebSocket, predicate?: (msg: any) => boolean): Promise<any> {
  return new Promise((resolve) => {
    const handler = (data: Buffer) => {
      const parsed = JSON.parse(data.toString());
      if (!predicate || predicate(parsed)) {
        ws.removeListener("message", handler);
        resolve(parsed);
      }
    };
    ws.on("message", handler);
  });
}

function waitForOpen(ws: WebSocket): Promise<void> {
  return new Promise((resolve) => {
    if (ws.readyState === WebSocket.OPEN) {
      resolve();
    } else {
      ws.on("open", () => resolve());
    }
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("WebSocket Server - Snapshot", () => {
  let diagram: DiagramCore;
  let wss: ReturnType<typeof createWSServer>;
  let port: number;

  beforeAll(async () => {
    port = await findAvailablePort();
    diagram = createDiagram();
    wss = createWSServer(diagram, diagram.events, { port });
  });

  afterAll(() => {
    wss._shutdown();
  });

  it("does not send automatic snapshot on connect", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);

    // Wait a short time — no message should arrive automatically
    const gotMessage = await Promise.race([
      waitForMessage(ws).then(() => true),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 300)),
    ]);

    expect(gotMessage).toBe(false);

    // But request_snapshot still works
    await waitForOpen(ws);
    ws.send(JSON.stringify({ type: "request_snapshot" }));
    const msg: WSSnapshotMessage = await waitForMessage(ws);
    expect(msg.type).toBe("snapshot");
    expect(msg.seq).toBeTypeOf("number");
    expect(Array.isArray(msg.nodes)).toBe(true);
    expect(Array.isArray(msg.edges)).toBe(true);

    ws.close();
  });

  it("request_snapshot returns current diagram state", async () => {
    // Add a node before connecting
    const stereo = diagram.getStereotype("Linear")!;
    diagram.addModule(stereo, 100, 50);

    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Request a snapshot explicitly
    ws.send(JSON.stringify({ type: "request_snapshot" }));

    const msg: WSSnapshotMessage = await waitForMessage(ws);

    expect(msg.nodes).toHaveLength(1);
    expect(msg.nodes[0].data.stereotype).toBe("Linear");

    ws.close();
  });

  it("supports request_snapshot to get fresh state", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Request a fresh snapshot
    ws.send(JSON.stringify({ type: "request_snapshot" }));

    const msg: WSSnapshotMessage = await waitForMessage(ws);

    expect(msg.type).toBe("snapshot");
    expect(Array.isArray(msg.nodes)).toBe(true);

    ws.close();
  });

  it("push_state sends snapshot back with imported state", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Send a push_state with test nodes/edges
    ws.send(JSON.stringify({
      type: "push_state",
      nodes: [{
        id: "test_1",
        type: "custom",
        position: { x: 100, y: 100 },
        data: { stereotype: "Input", label: "Input", params: {} },
      }],
      edges: [],
    }));

    const msg: WSSnapshotMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "snapshot",
    );

    expect(msg.type).toBe("snapshot");
    expect(msg.nodes).toHaveLength(1);
    expect(msg.nodes[0].id).toBe("test_1");
    expect(Array.isArray(msg.edges)).toBe(true);

    ws.close();
  });
});

describe("WebSocket Server - Delta Broadcast", () => {
  let diagram: DiagramCore;
  let wss: ReturnType<typeof createWSServer>;
  let port: number;

  beforeAll(async () => {
    port = await findAvailablePort();
    diagram = createDiagram();
    wss = createWSServer(diagram, diagram.events, { port });
  });

  afterAll(() => {
    wss._shutdown();
  });

  it("broadcasts delta when a module node is created", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Create a ReLU module node (triggers node_created event)
    const stereo = diagram.getStereotype("ReLU")!;
    diagram.addModule(stereo, 50, 50);

    const deltaMsg: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta"
    );

    expect(deltaMsg.type).toBe("delta");
    expect(deltaMsg.seq).toBeGreaterThanOrEqual(1);
    expect(deltaMsg.operations.length).toBeGreaterThan(0);
    expect(deltaMsg.operations[0].op).toBe("node_added");

    ws.close();
  });

  it("broadcasts delta when a join node is created", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Create an Addition join node
    const stereo = diagram.getStereotype("Addition")!;
    diagram.addJoinNode(stereo, 100, 100);

    const deltaMsg: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta"
    );

    expect(deltaMsg.type).toBe("delta");
    expect(deltaMsg.operations[0].op).toBe("node_added");

    ws.close();
  });

  it("broadcasts delta when an edge is created", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Create two nodes and connect them
    const reluStereo = diagram.getStereotype("ReLU")!;
    const tanhStereo = diagram.getStereotype("Tanh")!;
    diagram.addModule(reluStereo, 100, 100);
    diagram.addModule(tanhStereo, 100, 200);

    // Consume the node_created deltas (2 messages)
    await waitForMessage(ws, (msg) => msg.type === "delta");

    // Now find the node IDs
    const reluNode = diagram.nodes.find((n) => n.data.stereotype === "ReLU")!;
    const tanhNode = diagram.nodes.find((n) => n.data.stereotype === "Tanh")!;

    // Add edge between them
    diagram.addEdge(reluNode.id, tanhNode.id);

    const edgeDelta: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta" && msg.operations[0]?.op === "edge_added"
    );

    expect(edgeDelta.operations[0].op).toBe("edge_added");
    expect(edgeDelta.operations[0].edgeId).toBeTruthy();

    ws.close();
  });

  it("broadcasts delta when a node is deleted", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Create a node, then delete it
    const stereo = diagram.getStereotype("Dropout")!;
    diagram.addModule(stereo, 200, 200);

    // Consume node_created delta
    await waitForMessage(ws, (msg) => msg.type === "delta");

    const nodeId = diagram.nodes[diagram.nodes.length - 1].id;
    diagram.deleteNode(nodeId);

    const deltaMsg: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta" && msg.operations[0]?.op === "node_removed"
    );

    expect(deltaMsg.operations[0].op).toBe("node_removed");
    expect(deltaMsg.operations[0].nodeId).toBe(nodeId);

    ws.close();
  });

  it("broadcasts delta with correct seq numbering", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Create two nodes in sequence
    const linearStereo = diagram.getStereotype("Linear")!;
    diagram.addModule(linearStereo, 300, 100);
    const delta1: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta"
    );
    expect(delta1.seq).toBeGreaterThanOrEqual(1);

    diagram.addModule(linearStereo, 300, 200);
    const delta2: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta" && msg.seq !== delta1.seq
    );
    expect(delta2.seq).toBeGreaterThan(delta1.seq);

    ws.close();
  });

  it("snapshot seq and next delta seq are consecutive", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Send push_state to trigger a snapshot response
    ws.send(JSON.stringify({
      type: "push_state",
      nodes: [],
      edges: [],
    }));

    const snap: WSSnapshotMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "snapshot",
    );
    expect(snap.type).toBe("snapshot");
    const snapshotSeq = snap.seq;
    expect(snapshotSeq).toBeTypeOf("number");

    // Create a node — the delta seq should be exactly snapshotSeq + 1
    const stereo = diagram.getStereotype("Linear")!;
    diagram.addModule(stereo, 500, 100);

    const delta: WSDeltaMessage = await waitForMessage(
      ws,
      (msg) => msg.type === "delta"
    );

    expect(delta.seq).toBe(snapshotSeq + 1);
    expect(delta.operations[0].op).toBe("node_added");

    ws.close();
  });

  it("sends snapshots to multiple connected clients", async () => {
    const ws1 = new WebSocket(`ws://localhost:${port}`);
    const ws2 = new WebSocket(`ws://localhost:${port}`);

    // Wait for both to connect before sending
    await Promise.all([waitForOpen(ws1), waitForOpen(ws2)]);

    // Both request snapshots
    ws1.send(JSON.stringify({ type: "request_snapshot" }));
    ws2.send(JSON.stringify({ type: "request_snapshot" }));

    const [snap1, snap2] = await Promise.all([
      waitForMessage(ws1),
      waitForMessage(ws2),
    ]);

    expect(snap1.type).toBe("snapshot");
    expect(snap2.type).toBe("snapshot");

    // Create a node and verify both clients receive the delta
    const stereo = diagram.getStereotype("Sigmoid")!;
    diagram.addModule(stereo, 400, 100);

    const [delta1, delta2] = await Promise.all([
      waitForMessage(ws1, (msg) => msg.type === "delta"),
      waitForMessage(ws2, (msg) => msg.type === "delta"),
    ]);

    expect(delta1.operations[0].op).toBe("node_added");
    expect(delta2.operations[0].op).toBe("node_added");
    expect(delta1.seq).toBe(delta2.seq);

    ws1.close();
    ws2.close();
  });
});

describe("WebSocket Server - Error Handling", () => {
  let diagram: DiagramCore;
  let wss: ReturnType<typeof createWSServer>;
  let port: number;

  beforeAll(async () => {
    port = await findAvailablePort();
    diagram = createDiagram();
    wss = createWSServer(diagram, diagram.events, { port });
  });

  afterAll(() => {
    wss._shutdown();
  });

  it("ignores malformed client messages without crashing", async () => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws);

    // Send malformed messages
    ws.send("not valid json");
    ws.send("");
    ws.send('{"type": "unknown"}');

    // Wait a bit and verify server is still responsive
    await new Promise((resolve) => setTimeout(resolve, 200));

    // Server should still respond to request_snapshot
    const ws2 = new WebSocket(`ws://localhost:${port}`);
    await waitForOpen(ws2);
    ws2.send(JSON.stringify({ type: "request_snapshot" }));
    const msg: WSSnapshotMessage = await waitForMessage(ws2);
    expect(msg.type).toBe("snapshot");

    ws.close();
    ws2.close();
  });

  it("cannot connect to a closed server", async () => {
    // Create a separate server on a new port that we'll close
    const tempPort = await findAvailablePort();
    const tempDiagram = createDiagram();
    const tempWss = createWSServer(tempDiagram, tempDiagram.events, { port: tempPort });

    // Close it immediately
    tempWss._shutdown();

    // Give it a moment to fully close
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Attempting to connect should fail
    await expect(
      new Promise<void>((resolve, reject) => {
        const ws = new WebSocket(`ws://localhost:${tempPort}`);
        ws.on("open", () => {
          ws.close();
          reject(new Error("Should not have connected"));
        });
        ws.on("error", () => resolve());
        setTimeout(() => reject(new Error("Connection timed out")), 2000);
      })
    ).resolves.toBeUndefined();
  });
});
