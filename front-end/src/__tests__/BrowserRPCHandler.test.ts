import { describe, it, expect, vi, afterAll } from "vitest";
import { BrowserRPCHandler } from "../sync/BrowserRPCHandler";
import { Diagram } from "../Diagram.svelte";
import { stubWindow, unstubWindow } from "./helpers";
import type { Node, Edge } from "@xyflow/svelte";

stubWindow();
afterAll(() => unstubWindow());

describe("BrowserRPCHandler", () => {
  /**
   * Create a BrowserRPCHandler with a real Diagram but stub the WebSocket
   * so we can inspect outgoing RPC responses without a real connection.
   */
  function createHandler() {
    const diagram = new Diagram();
    // Reset to known state (Diagram auto-spawns an Input node, which we clear)
    diagram.nodes = [];
    diagram.edges = [];

    const handler = new BrowserRPCHandler(diagram, "ws://localhost:0");

    // Stub the internal WebSocket with a mock
    const mockSend = vi.fn();
    (handler as any).ws = {
      send: mockSend,
      readyState: WebSocket.OPEN,
      close: vi.fn(),
    };

    return { handler, diagram, mockSend };
  }

  // ── Dispatch to correct method ──────────────────────────────────────

  it("dispatches get_graph and returns nodes and edges", () => {
    const { handler, diagram, mockSend } = createHandler();

    // Attach some test data
    const testNode = { id: "n1", type: "custom", position: { x: 0, y: 0 }, data: {} } as Node;
    const testEdge = { id: "e1", source: "n1", target: "n2" } as Edge;
    diagram.nodes = [testNode];
    diagram.edges = [testEdge];

    // Simulate an incoming RPC request via WebSocket
    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-1", method: "get_graph" }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);

    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-1");
    expect(response.result).toBeDefined();
    expect(response.result.nodes).toHaveLength(1);
    expect(response.result.nodes[0].id).toBe("n1");
    expect(response.result.edges).toHaveLength(1);
    expect(response.result.edges[0].id).toBe("e1");
  });

  it("dispatches get_node and returns the matching node", () => {
    const { handler, diagram, mockSend } = createHandler();

    const testNode = { id: "n42", type: "custom", position: { x: 10, y: 20 }, data: { name: "TestNode" } } as Node;
    diagram.nodes = [testNode];

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-2", method: "get_node", params: { nodeId: "n42" } }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);
    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-2");
    expect(response.result).toBeDefined();
    expect(response.result.id).toBe("n42");
  });

  it("dispatches ping and returns status ok", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-ping", method: "ping" }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);
    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-ping");
    expect(response.result).toBeDefined();
    expect(response.result.status).toBe("ok");
  });

  // ── Unknown method returns error ────────────────────────────────────

  it("returns error for unknown method", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-unknown", method: "nonexistent" }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);

    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-unknown");
    expect(response.error).toBeDefined();
    expect(response.error.message).toContain("Unknown method");
  });

  it("returns error for unknown method with params", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-99", method: "fly_to_moon", params: { fuel: "hydrogen" } }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);
    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-99");
    expect(response.error).toBeDefined();
    expect(response.error.message).toContain("Unknown method");
  });

  // ── Error in handler returns error response ─────────────────────────

  it("returns error when handler throws (missing required parameter)", () => {
    const { handler, mockSend } = createHandler();

    // delete_nodes without nodeIds param — handler throws before mutating
    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-3", method: "delete_nodes", params: {} }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);

    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-3");
    expect(response.error).toBeDefined();
    expect(response.error.message).toBe("nodeIds must be an array");
  });

  it("returns error when get_node is called without nodeId", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-4", method: "get_node", params: {} }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);
    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-4");
    expect(response.error).toBeDefined();
    expect(response.error.message).toContain("Missing required parameter");
  });

  it("returns error when get_node targets nonexistent node", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ id: "req-5", method: "get_node", params: { nodeId: "ghost" } }),
    });

    expect(mockSend).toHaveBeenCalledTimes(1);
    const response = JSON.parse(mockSend.mock.calls[0][0]);
    expect(response.id).toBe("req-5");
    expect(response.error).toBeDefined();
    expect(response.error.message).toContain("Node not found");
  });

  // ── Edge cases: malformed messages ──────────────────────────────────

  it("ignores invalid JSON in messages", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: "not valid json",
    });

    // No response should be sent for unparseable messages
    expect(mockSend).not.toHaveBeenCalled();
  });

  it("ignores malformed RPC requests (missing id and method)", () => {
    const { handler, mockSend } = createHandler();

    (handler as any).handleMessage({
      data: JSON.stringify({ foo: "bar" }),
    });

    expect(mockSend).not.toHaveBeenCalled();
  });
});
