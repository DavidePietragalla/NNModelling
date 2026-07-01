// mcp-server/src/browser-client.ts
// WebSocket RPC client — listens for browser connections, provides
// promise-based request/response (not delta broadcast).

import { WebSocketServer, WebSocket } from "ws";

interface RPCPending {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export interface BrowserRPCClientConfig {
  host?: string;
  port?: number;
  connectionTimeout?: number;
  requestTimeout?: number;
}

/**
 * BrowserRPCClient — WebSocket server that accepts browser connections
 * and provides a promise-based RPC interface.
 *
 * The browser connects TO this server. When an MCP tool is called via stdio,
 * the server sends an RPC request to the browser over the WebSocket connection.
 *
 * Protocol:
 *   Request:  {id: string, method: string, params: object}
 *   Response: {id: string, result: any}
 *   Error:    {id: string, error: {message: string}}
 */
export class BrowserRPCClient {
  private wss: WebSocketServer | null = null;
  private client: WebSocket | null = null;
  private pending = new Map<string, RPCPending>();
  private nextId = 0;
  private host: string;
  private port: number;
  private connectionTimeout: number;
  private requestTimeout: number;

  constructor(config?: BrowserRPCClientConfig) {
    this.host = config?.host ?? "localhost";
    this.port = config?.port ?? 9339;
    this.connectionTimeout = config?.connectionTimeout ?? 60000;
    this.requestTimeout = config?.requestTimeout ?? 30000;
  }

  /** Start the WebSocket server and wait for a browser connection. */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.wss = new WebSocketServer({ host: this.host, port: this.port });

      this.wss.on("connection", (ws: WebSocket) => {
        if (this.client) {
          // Only one browser connection at a time
          ws.close(1013, "Only one browser connection allowed");
          return;
        }

        this.client = ws;
        console.error(
          `[browser-client] Browser connected on ws://${this.host}:${this.port}`,
        );

        ws.on("message", (data: Buffer) => {
          this.onMessage(data.toString());
        });

        ws.on("close", () => {
          console.error("[browser-client] Browser disconnected");
          this.client = null;
          this.rejectAll(new Error("Browser disconnected"));
        });

        ws.on("error", (err: Error) => {
          console.error("[browser-client] WebSocket error:", err.message);
        });

        resolve();
      });

      this.wss.on("error", (err: Error) => {
        reject(err);
      });

      console.error(
        `[browser-client] Listening on ws://${this.host}:${this.port}`,
      );

      // Timeout waiting for browser connection
      setTimeout(() => {
        if (!this.client) {
          reject(new Error("Timeout waiting for browser connection"));
        }
      }, this.connectionTimeout);
    });
  }

  /** Close the WebSocket server and reject all pending requests. */
  close(): void {
    this.rejectAll(new Error("Server shutting down"));
    this.client?.close();
    this.client = null;
    this.wss?.close();
    this.wss = null;
  }

  /**
   * Send an RPC request to the browser and await the response.
   * @param method  The RPC method name (e.g. "get_graph", "create_node").
   * @param params  Optional parameters for the method.
   * @returns The browser's response (decoded from JSON).
   */
  call<T = unknown>(method: string, params?: unknown): Promise<T> {
    if (!this.client || this.client.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("No browser connected"));
    }

    const id = String(++this.nextId);

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`RPC timeout: ${method}`));
      }, this.requestTimeout);

      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer,
      });

      this.client!.send(JSON.stringify({ id, method, params: params ?? {} }));
    });
  }

  /** Check if a browser is currently connected. */
  isConnected(): boolean {
    return this.client !== null && this.client.readyState === WebSocket.OPEN;
  }

  // ── Private ────────────────────────────────────────────────────────

  private onMessage(data: string): void {
    let msg: { id?: string; result?: unknown; error?: { message: string } };
    try {
      msg = JSON.parse(data);
    } catch {
      return; // Ignore non-JSON messages
    }

    if (!msg.id) return;

    const pending = this.pending.get(msg.id);
    if (!pending) return;

    clearTimeout(pending.timer);
    this.pending.delete(msg.id);

    if (msg.error) {
      pending.reject(new Error(msg.error.message));
    } else {
      pending.resolve(msg.result);
    }
  }

  private rejectAll(err: Error): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }
}
