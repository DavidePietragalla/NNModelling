/**
 * BrowserRPCClient Tests
 *
 * Tests the WebSocket RPC client that communicates with the browser.
 * Uses real WebSocket connections on dynamic ports.
 *
 * Test scenarios:
 *   1. connect() waits for browser connection
 *   2. call() sends {id, method, params} and receives {id, result}
 *   3. call() rejects on timeout
 *   4. call() rejects on {id, error: {message}}
 *   5. close() shuts down server
 *   6. connect() rejects on port conflict
 *   7. Multiple sequential calls work correctly
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { BrowserRPCClient } from "../src/browser-client";
import { WebSocketServer, WebSocket } from "ws";

// ── Helpers ───────────────────────────────────────────────────────────────

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

/**
 * Simulates a browser connecting to the BrowserRPCClient's WebSocket server.
 * Returns the WebSocket that the browser would use, plus a helper to
 * simulate sending responses.
 */
function createBrowserConnection(
  port: number,
): Promise<{ ws: WebSocket; respond: (msg: object) => void }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://localhost:${port}`);
    ws.on("open", () => {
      resolve({
        ws,
        respond: (msg: object) => ws.send(JSON.stringify(msg)),
      });
    });
    ws.on("error", reject);
    setTimeout(() => reject(new Error("Connection timeout")), 3000);
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("BrowserRPCClient - connect", () => {
  it("connect() resolves when browser connects", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port, connectionTimeout: 5000 });

    // Start listening and connect a simulated browser in parallel
    const connectPromise = client.connect();
    const browserConn = await createBrowserConnection(port);

    await expect(connectPromise).resolves.toBeUndefined();
    expect(client.isConnected()).toBe(true);

    browserConn.ws.close();
    client.close();
  });

  it("connect() rejects on timeout when no browser connects", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port, connectionTimeout: 500 });

    await expect(client.connect()).rejects.toThrow("Timeout waiting for browser connection");

    client.close();
  });
});

describe("BrowserRPCClient - call", () => {
  let port: number;
  let client: BrowserRPCClient;
  let browser: { ws: WebSocket; respond: (msg: object) => void };

  beforeAll(async () => {
    port = await findAvailablePort();
    client = new BrowserRPCClient({ port, connectionTimeout: 5000, requestTimeout: 5000 });

    // Connect browser before running tests
    const connectPromise = client.connect();
    browser = await createBrowserConnection(port);
    await connectPromise;
  });

  afterAll(() => {
    browser.ws.close();
    client.close();
  });

  it("call() sends {id, method, params} and resolves with result", async () => {
    const callPromise = client.call("get_graph", {});

    // Browser receives the request
    const message = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });

    const parsed = JSON.parse(message);
    expect(parsed).toHaveProperty("id");
    expect(parsed.method).toBe("get_graph");
    expect(parsed.params).toEqual({});

    // Browser responds
    browser.respond({ id: parsed.id, result: { nodes: [], edges: [] } });

    const result = await callPromise;
    expect(result).toEqual({ nodes: [], edges: [] });
  });

  it("call() sends correct method name and params", async () => {
    const callPromise = client.call("create_node", { stereotype: "Linear", position: { x: 100, y: 50 } });

    const message = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });

    const parsed = JSON.parse(message);
    expect(parsed.method).toBe("create_node");
    expect(parsed.params).toEqual({ stereotype: "Linear", position: { x: 100, y: 50 } });

    browser.respond({ id: parsed.id, result: { nodeId: "n1" } });
    await callPromise;
  });

  it("call() rejects on timeout", async () => {
    const timeoutPort = await findAvailablePort();
    const shortTimeoutClient = new BrowserRPCClient({ port: timeoutPort, connectionTimeout: 1000, requestTimeout: 100 });
    const shortConnectPromise = shortTimeoutClient.connect();
    const shortBrowser = await createBrowserConnection(timeoutPort);
    await shortConnectPromise;

    // Make a call but don't respond — it should time out
    await expect(shortTimeoutClient.call("get_graph", {})).rejects.toThrow("RPC timeout");

    shortBrowser.ws.close();
    shortTimeoutClient.close();
  });

  it("call() rejects on {id, error: {message}}", async () => {
    const callPromise = client.call("get_node", { nodeId: "nonexistent" });

    const message = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });

    const parsed = JSON.parse(message);

    // Browser responds with error
    browser.respond({ id: parsed.id, error: { message: "Node 'nonexistent' not found" } });

    await expect(callPromise).rejects.toThrow("Node 'nonexistent' not found");
  });

  it("multiple sequential calls work correctly", async () => {
    // First call
    const call1Promise = client.call("ping", {});
    const msg1 = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });
    const parsed1 = JSON.parse(msg1);
    expect(parsed1.method).toBe("ping");
    browser.respond({ id: parsed1.id, result: { status: "ok" } });
    const result1 = await call1Promise;
    expect(result1).toEqual({ status: "ok" });

    // Second call (different method)
    const call2Promise = client.call("validate_graph", {});
    const msg2 = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });
    const parsed2 = JSON.parse(msg2);
    expect(parsed2.method).toBe("validate_graph");
    browser.respond({ id: parsed2.id, result: { valid: true, errors: [] } });
    const result2 = await call2Promise;
    expect(result2).toEqual({ valid: true, errors: [] });
  });
});

describe("BrowserRPCClient - close", () => {
  it("close() shuts down server and rejects pending calls", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port, connectionTimeout: 5000 });
    const connectPromise = client.connect();
    const browser = await createBrowserConnection(port);
    await connectPromise;

    // Make a call that will be pending when we close
    const callPromise = client.call("get_graph", {});

    // Read (and discard) the message so it doesn't interfere with later tests
    browser.ws.once("message", () => {});

    // Close the server
    client.close();

    // Pending call should reject
    await expect(callPromise).rejects.toThrow("Server shutting down");

    browser.ws.close();
  });

  it("isConnected returns false after close", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port, connectionTimeout: 5000 });
    const connectPromise = client.connect();
    const browser = await createBrowserConnection(port);
    await connectPromise;

    expect(client.isConnected()).toBe(true);

    client.close();
    expect(client.isConnected()).toBe(false);

    browser.ws.close();
  });
});
