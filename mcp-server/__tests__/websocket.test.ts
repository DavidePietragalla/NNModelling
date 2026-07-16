/**
 * BrowserRPCClient Tests (Multi-Tab)
 *
 * Tests the WebSocket RPC client that communicates with browser tabs.
 * Uses real WebSocket connections on dynamic ports.
 *
 * Key differences from old API:
 *   - start() is non-blocking (returns void, not Promise)
 *   - Multiple tabs supported with auto-selection for first tab
 *   - call() sends RPC to the active tab
 *   - Initial ping is sent to each tab on connection
 *
 * Test scenarios:
 *   1. start() opens port and accepts connections
 *   2. call() sends {id, method, params} and receives {id, result}
 *   3. call() rejects on timeout
 *   4. call() rejects on {id, error: {message}}
 *   5. close() shuts down server and rejects pending calls
 *   6. isConnected() reflects active tab state
 *   7. Multiple sequential calls work correctly
 *   8. Multiple tabs + tab selection
 *   9. Tab disconnection removes it from tab list
 *   10. getTabs() returns TabInfo with nodeCount/edgeCount after ping
 *   11. selectTab() throws for unknown tab id
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { BrowserRPCClient } from "../src/browser-client";
import { WebSocket } from "ws";

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
 * Simulates a browser tab connecting to the BrowserRPCClient's WebSocket server.
 * Automatically handles the initial ping (sends {nodeCount, edgeCount} response).
 * Returns the WebSocket and a helper to send JSON responses.
 */
function createBrowserConnection(
  port: number,
): Promise<{ ws: WebSocket; respond: (msg: object) => void }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://localhost:${port}`);

    ws.on("open", async () => {
      // Read and respond to the initial ping the server sends on connection
      try {
        const firstMsg = await new Promise<string>((res) => {
          ws.once("message", (d: Buffer) => res(d.toString()));
          // Safety timeout — if no ping arrives within 2s, resolve anyway
          setTimeout(() => res(""), 2000);
        });

        if (firstMsg) {
          const parsed = JSON.parse(firstMsg);
          if (parsed.method === "ping") {
            ws.send(
              JSON.stringify({
                id: parsed.id,
                result: { nodeCount: 0, edgeCount: 0 },
              }),
            );
          }
        }
      } catch {
        // Ignore ping handling errors
      }

      resolve({
        ws,
        respond: (msg: object) => ws.send(JSON.stringify(msg)),
      });
    });

    ws.on("error", reject);
    setTimeout(() => reject(new Error("Connection timeout")), 5000);
  });
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("BrowserRPCClient - start", () => {
  it("start() opens port and browser can connect", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });

    expect(client.isConnected()).toBe(false);
    expect(client.getTabs()).toEqual([]);
    expect(client.getActiveTabId()).toBeNull();

    client.start();

    const browser = await createBrowserConnection(port);

    expect(client.isConnected()).toBe(true);
    expect(client.getActiveTabId()).toBe("tab_1");
    expect(client.getTabs()).toHaveLength(1);
    expect(client.getTabs()[0].id).toBe("tab_1");

    browser.ws.close();
    client.close();
  });

  it("start() is idempotent (multiple calls do not create multiple servers)", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();
    client.start(); // Second start should be a no-op
    client.start(); // Third start should be a no-op

    const browser = await createBrowserConnection(port);
    expect(client.isConnected()).toBe(true);

    browser.ws.close();
    client.close();
  });
});

describe("BrowserRPCClient - call", () => {
  let port: number;
  let client: BrowserRPCClient;
  let browser: { ws: WebSocket; respond: (msg: object) => void };

  beforeAll(async () => {
    port = await findAvailablePort();
    client = new BrowserRPCClient({
      port,
      requestTimeout: 5000,
    });
    client.start();
    browser = await createBrowserConnection(port);
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
    const callPromise = client.call("create_node", {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
    });

    const message = await new Promise<string>((resolve) => {
      browser.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });

    const parsed = JSON.parse(message);
    expect(parsed.method).toBe("create_node");
    expect(parsed.params).toEqual({
      stereotype: "Linear",
      position: { x: 100, y: 50 },
    });

    browser.respond({ id: parsed.id, result: { nodeId: "n1" } });
    await callPromise;
  });

  it("call() rejects on timeout", async () => {
    const timeoutPort = await findAvailablePort();
    const shortTimeoutClient = new BrowserRPCClient({
      port: timeoutPort,
      requestTimeout: 100,
    });
    shortTimeoutClient.start();
    const shortBrowser = await createBrowserConnection(timeoutPort);

    // Make a call but don't respond — it should time out
    await expect(
      shortTimeoutClient.call("get_graph", {}),
    ).rejects.toThrow("RPC timeout");

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
    browser.respond({
      id: parsed.id,
      error: { message: "Node 'nonexistent' not found" },
    });

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
    browser.respond({
      id: parsed2.id,
      result: { valid: true, errors: [] },
    });
    const result2 = await call2Promise;
    expect(result2).toEqual({ valid: true, errors: [] });
  });

  it("call() rejects when no active tab", async () => {
    const isolatedPort = await findAvailablePort();
    const isolatedClient = new BrowserRPCClient({ port: isolatedPort });
    isolatedClient.start();
    // No browser connected — call should reject
    await expect(isolatedClient.call("get_graph", {})).rejects.toThrow(
      "No browser connected",
    );
    isolatedClient.close();
  });
});

describe("BrowserRPCClient - multi-tab", () => {
  it("supports multiple browser tabs", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();

    // First tab connects — auto-selected
    const tab1 = await createBrowserConnection(port);
    expect(client.getActiveTabId()).toBe("tab_1");
    expect(client.getTabs()).toHaveLength(1);

    // Second tab connects — NOT auto-selected
    const tab2 = await createBrowserConnection(port);
    expect(client.getActiveTabId()).toBe("tab_1"); // Still tab_1
    expect(client.getTabs()).toHaveLength(2);

    // Check tab info
    const tabs = client.getTabs();
    expect(tabs[0].id).toBe("tab_1");
    expect(tabs[1].id).toBe("tab_2");
    expect(typeof tabs[0].connectedAt).toBe("number");
    expect(typeof tabs[1].connectedAt).toBe("number");

    // Select tab_2
    client.selectTab("tab_2");
    expect(client.getActiveTabId()).toBe("tab_2");

    // call() should now go to tab_2
    const callPromise = client.call("list_stereotypes", {});
    const msg = await new Promise<string>((resolve) => {
      tab2.ws.once("message", (data: Buffer) => resolve(data.toString()));
    });
    const parsed = JSON.parse(msg);
    expect(parsed.method).toBe("list_stereotypes");

    tab2.respond({
      id: parsed.id,
      result: { stereotypes: [] },
    });
    await expect(callPromise).resolves.toEqual({ stereotypes: [] });

    client.close();
    tab1.ws.close();
    tab2.ws.close();
  });

  it("tab disconnection removes it from tab list", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();

    const tab1 = await createBrowserConnection(port);
    const tab2 = await createBrowserConnection(port);
    expect(client.getTabs()).toHaveLength(2);

    // Disconnect tab_1
    tab1.ws.close();

    // Give time for close to be processed
    await new Promise((resolve) => setTimeout(resolve, 100));

    const tabs = client.getTabs();
    expect(tabs).toHaveLength(1);
    expect(tabs[0].id).toBe("tab_2");

    client.close();
    tab2.ws.close();
  });

  it("disconnecting active tab clears activeTabId", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();

    const tab = await createBrowserConnection(port);
    expect(client.getActiveTabId()).toBe("tab_1");

    tab.ws.close();

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(client.getActiveTabId()).toBeNull();
    expect(client.isConnected()).toBe(false);

    client.close();
  });

  it("selectTab() throws for unknown tab id", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();
    // No tabs connected — any selectTab should throw
    expect(() => client.selectTab("nonexistent")).toThrow("not found");
    client.close();
  });

  it("getTabs() returns TabInfo with nodeCount/edgeCount after ping", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port, requestTimeout: 5000 });
    client.start();

    // Connect a browser that responds to pings with custom stats.
    // Register message handler BEFORE awaiting open to avoid race.
    const ws = new WebSocket(`ws://localhost:${port}`);
    const firstMsg = new Promise<string>((resolve) => {
      ws.once("message", (data: Buffer) => resolve(data.toString()));
    });
    await new Promise<void>((resolve, reject) => {
      ws.on("open", resolve);
      ws.on("error", reject);
    });

    // Wait for and respond to the ping
    const msg = await firstMsg;
    const parsed = JSON.parse(msg);
    expect(parsed.method).toBe("ping");
    ws.send(
      JSON.stringify({
        id: parsed.id,
        result: { nodeCount: 5, edgeCount: 4 },
      }),
    );

    // Give time for response to be processed
    await new Promise((resolve) => setTimeout(resolve, 100));

    const tabs = client.getTabs();
    expect(tabs).toHaveLength(1);
    expect(tabs[0].nodeCount).toBe(5);
    expect(tabs[0].edgeCount).toBe(4);

    ws.close();
    client.close();
  });
});

describe("BrowserRPCClient - close", () => {
  it("close() releases the listening port before resolving", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });

    await client.start();
    await client.close();

    const replacement = new BrowserRPCClient({ port });
    await expect(replacement.start()).resolves.toBeUndefined();
    await replacement.close();
  });

  it("close() shuts down server and rejects pending calls", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({
      port,
      requestTimeout: 5000,
    });
    client.start();
    const browser = await createBrowserConnection(port);

    // Make a call that will be pending when we close
    const callPromise = client.call("get_graph", {});

    // Read (and discard) the message so it doesn't interfere
    browser.ws.once("message", () => {});

    // Close the server
    client.close();

    // Pending call should reject
    await expect(callPromise).rejects.toThrow("Server shutting down");

    browser.ws.close();
  });

  it("isConnected returns false after close", async () => {
    const port = await findAvailablePort();
    const client = new BrowserRPCClient({ port });
    client.start();
    const browser = await createBrowserConnection(port);

    expect(client.isConnected()).toBe(true);

    client.close();
    expect(client.isConnected()).toBe(false);

    browser.ws.close();
  });
});
