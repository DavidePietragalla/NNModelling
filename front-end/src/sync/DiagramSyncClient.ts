// front-end/src/sync/DiagramSyncClient.ts
// Browser-side WebSocket sync client.
// Receives delta messages from the MCP server's WebSocket broadcaster
// and applies them to the reactive $state.raw arrays in Diagram.
//
// Protocol:
//   Server → Client:  WSSnapshotMessage | WSDeltaMessage
//   Client → Server:  { type: "request_snapshot" }  (on sequence gap)

import type { Diagram } from "../Diagram.svelte";
import type {
  WSDeltaMessage,
  WSSnapshotMessage,
  DeltaOperation,
} from "../core/types";
import type { Node, Edge } from "@xyflow/svelte";

export class DiagramSyncClient {
  private ws: WebSocket | null = null;
  private lastSeenSeq: number = 0;
  private diagram: Diagram;
  private url: string;
  private reconnectDelay: number = 1000;
  private intentionalClose: boolean = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * @param diagram  The Diagram instance to mutate.
   * @param url      WebSocket URL. Defaults to /ws in dev (proxied via Vite),
   *                 or ws://localhost:9339 in production.
   */
  constructor(diagram: Diagram, url?: string) {
    this.diagram = diagram;
    this.url =
      url ??
      (import.meta.env.DEV
        ? `ws://${window.location.host}/ws`
        : `ws://localhost:9339`);
  }

  // ── Public API ───────────────────────────────────────────────────

  /** Open the WebSocket connection and start receiving updates. */
  connect(): void {
    this.intentionalClose = false;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.debug(`[SyncClient] Connected to ${this.url}`);
      this.reconnectDelay = 1000; // Reset exponential backoff
      // Push browser state to server so MCP tools reflect the canvas
      this.pushState();
    };

    this.ws.onmessage = (event: MessageEvent) => {
      this.handleMessage(event);
    };

    this.ws.onclose = () => {
      if (!this.intentionalClose) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = (event: Event) => {
      console.error("[SyncClient] WebSocket error:", event);
    };
  }

  /** Close the WebSocket connection and stop reconnecting. */
  disconnect(): void {
    this.intentionalClose = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  // ── Message Handling ─────────────────────────────────────────────

  private handleMessage(event: MessageEvent): void {
    let msg: unknown;
    try {
      msg = JSON.parse(event.data as string);
    } catch {
      console.error("[SyncClient] Failed to parse WebSocket message:", event.data);
      return;
    }

    if (!isSnapshotMessage(msg) && !isDeltaMessage(msg)) {
      console.warn("[SyncClient] Unknown message type:", msg);
      return;
    }

    if (msg.type === "snapshot") {
      this.applySnapshot(msg);
    } else {
      this.applyDelta(msg);
    }
  }

  private applySnapshot(msg: WSSnapshotMessage): void {
    console.debug(`[SyncClient] Applying snapshot (seq=${msg.seq}, ${msg.nodes.length} nodes, ${msg.edges.length} edges)`);
    this.diagram.nodes = msg.nodes;
    this.diagram.edges = msg.edges;
    this.lastSeenSeq = msg.seq;
  }

  private applyDelta(msg: WSDeltaMessage): void {
    // Sequence gap detection: if we missed events, request full snapshot
    if (this.lastSeenSeq > 0 && msg.seq !== this.lastSeenSeq + 1) {
      console.warn(
        `[SyncClient] Sequence gap: expected ${this.lastSeenSeq + 1}, got ${msg.seq}. Requesting snapshot.`,
      );
      this.requestSnapshot();
      return;
    }

    for (const op of msg.operations) {
      this.applyOperation(op);
    }
    this.lastSeenSeq = msg.seq;
  }

  private requestSnapshot(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "request_snapshot" }));
    }
  }

  /** Push the current browser canvas state to the MCP server. */
  private pushState(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "push_state",
          nodes: this.diagram.nodes,
          edges: this.diagram.edges,
        }),
      );
    }
  }

  // ── Operation Application ───────────────────────────────────────

  private applyOperation(op: DeltaOperation): void {
    switch (op.op) {
      case "node_added": {
        // Server guarantees that op.data contains all required Node fields
        const newNode = { id: op.nodeId, ...op.data } as Node;
        this.diagram.nodes = [...this.diagram.nodes, newNode];
        break;
      }

      case "node_removed": {
        this.diagram.nodes = this.diagram.nodes.filter(
          (n) => n.id !== op.nodeId,
        );
        break;
      }

      case "node_moved": {
        this.diagram.nodes = this.diagram.nodes.map((n) =>
          n.id === op.nodeId ? { ...n, position: op.position } : n,
        );
        break;
      }

      case "node_updated": {
        this.diagram.nodes = this.diagram.nodes.map((n) =>
          n.id === op.nodeId
            ? ({ ...n, data: { ...n.data, ...op.changes } } as Node)
            : n,
        );
        break;
      }

      case "edge_added": {
        const newEdge = { id: op.edgeId, ...op.data } as Edge;
        this.diagram.edges = [...this.diagram.edges, newEdge];
        break;
      }

      case "edge_removed": {
        this.diagram.edges = this.diagram.edges.filter(
          (e) => e.id !== op.edgeId,
        );
        break;
      }

      case "edge_reconnected": {
        this.diagram.edges = this.diagram.edges.map((e) =>
          e.id === op.edgeId ? ({ ...e, ...op.changes } as Edge) : e,
        );
        break;
      }

      case "selection_changed": {
        this.diagram.nodes = this.diagram.nodes.map((n) => ({
          ...n,
          selected: op.nodeIds.includes(n.id),
        }));
        this.diagram.edges = this.diagram.edges.map((e) => ({
          ...e,
          selected: op.edgeIds.includes(e.id),
        }));
        break;
      }

      case "graph_reset": {
        this.diagram.nodes = op.nodes;
        this.diagram.edges = op.edges;
        break;
      }
    }
  }

  // ── Reconnection ─────────────────────────────────────────────────

  private scheduleReconnect(): void {
    const delay = this.reconnectDelay;
    console.debug(`[SyncClient] Reconnecting in ${delay}ms...`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 30000);
      this.connect();
    }, delay);
  }
}

// ── Type Guards ────────────────────────────────────────────────────

function isSnapshotMessage(msg: unknown): msg is WSSnapshotMessage {
  return (
    typeof msg === "object" &&
    msg !== null &&
    (msg as Record<string, unknown>).type === "snapshot"
  );
}

function isDeltaMessage(msg: unknown): msg is WSDeltaMessage {
  return (
    typeof msg === "object" &&
    msg !== null &&
    (msg as Record<string, unknown>).type === "delta"
  );
}
