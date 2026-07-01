// mcp-server/src/ws-server.ts
// WebSocket delta broadcaster — subscribes to EventBus and sends incremental
// updates to connected browser clients. On connect, sends a full snapshot.

import { WebSocketServer, WebSocket } from "ws";
import type { EventBus } from "@nnmodelling/front-end/core/EventBus";
import type { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import type {
  DomainEvent,
  DeltaOperation,
  WSDeltaMessage,
  WSSnapshotMessage,
  Node,
  Edge,
} from "@nnmodelling/front-end/core/types";

export interface WSServerConfig {
  port: number;               // Default: 9339
  host?: string;              // Default: "localhost"
  maxClients?: number;        // Default: 10
  pingInterval?: number;      // Default: 30000 (30s)
}

/** Extended WebSocketServer with external shutdown trigger. */
export interface WSServer extends WebSocketServer {
  _shutdown: () => void;
}

export function createWSServer(
  diagram: DiagramCore,
  eventBus: EventBus,
  config: WSServerConfig,
): WSServer {
  const host = config.host ?? "localhost";
  const port = config.port;
  const maxClients = config.maxClients ?? 10;
  const pingInterval = config.pingInterval ?? 30000;

  const wss = new WebSocketServer({ port, host }) as WSServer;

  // Broadcast sequence counter — only increments when a message is actually
  // sent. This prevents seq gaps when events like graph_changed are skipped.
  let broadcastSeq = 0;

  console.error(
    `[nnmodelling-ws] WebSocket server listening on ws://${host}:${port}`,
  );

  // ── Ping interval to keep connections alive ────────────────
  let pingTimer: ReturnType<typeof setInterval> | null = null;

  // ── Connection handler ─────────────────────────────────────
  wss.on("connection", (ws: WebSocket) => {
    const totalClients = wss.clients.size;

    // Enforce max clients
    if (totalClients > maxClients) {
      console.error(
        `[nnmodelling-ws] Max clients (${maxClients}) reached. Rejecting new connection.`,
      );
      ws.close(1013, "Too many clients");
      return;
    }

    console.error(`[nnmodelling-ws] Browser connected (total: ${totalClients})`);

    // ── Handle client messages (e.g., request_snapshot, push_state) ──
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "request_snapshot") {
          const snap: WSSnapshotMessage = {
            type: "snapshot",
            seq: broadcastSeq,
            nodes: diagram.nodes,
            edges: diagram.edges,
          };
          ws.send(JSON.stringify(snap));
        } else if (msg.type === "push_state") {
          // Browser pushes its canvas state to the MCP server.
          // This imports the browser's diagram into the server's DiagramCore
          // so MCP tools (get_graph, etc.) reflect the current canvas.
          try {
            diagram.importFromJson(
              JSON.stringify({ nodes: msg.nodes, edges: msg.edges }),
            );
            console.error(
              `[nnmodelling-ws] Imported state from browser (${msg.nodes?.length ?? 0} nodes, ${msg.edges?.length ?? 0} edges)`,
            );
            // Send snapshot back to confirm the imported state
            broadcastSeq++;
            const snap: WSSnapshotMessage = {
              type: "snapshot",
              seq: broadcastSeq,
              nodes: diagram.nodes,
              edges: diagram.edges,
            };
            ws.send(JSON.stringify(snap));
          } catch (importErr) {
            console.error(
              "[nnmodelling-ws] Failed to import browser state:",
              importErr,
            );
          }
        }
      } catch {
        // Ignore malformed messages
      }
    });

    // ── Handle close ─────────────────────────────────────────
    ws.on("close", () => {
      console.error(
        `[nnmodelling-ws] Browser disconnected (total: ${wss.clients.size})`,
      );
    });

    // ── Handle errors ────────────────────────────────────────
    ws.on("error", (err: Error) => {
      console.error(`[nnmodelling-ws] WebSocket error:`, err.message);
    });
  });

  // ── Subscribe to EventBus and broadcast deltas ────────────
  const unsubscribe = eventBus.onAny((event: DomainEvent) => {
    const operations = domainEventToDeltaOps(event);
    if (operations.length === 0) return;

    broadcastSeq++;

    const delta: WSDeltaMessage = {
      type: "delta",
      seq: broadcastSeq,
      operations,
    };

    const payload = JSON.stringify(delta);
    for (const client of wss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(payload);
      }
    }
  });

  // ── Ping/pong keep-alive ──────────────────────────────────
  pingTimer = setInterval(() => {
    for (const client of wss.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.ping();
      }
    }
  }, pingInterval);

  // ── Graceful shutdown ─────────────────────────────────────
  const shutdown = () => {
    console.error("[nnmodelling-ws] Shutting down WebSocket server...");
    unsubscribe();
    if (pingTimer) clearInterval(pingTimer);
    wss.close(() => {
      console.error("[nnmodelling-ws] WebSocket server closed");
    });
  };

  // Allow external shutdown trigger
  wss._shutdown = shutdown;

  return wss;
}

/**
 * Convert a DomainEvent to an array of DeltaOperations.
 * Some events map to multiple operations (e.g., node_deleted also removes
 * attached edges).
 */
function domainEventToDeltaOps(event: DomainEvent): DeltaOperation[] {
  const p = event.payload as Record<string, unknown>;

  switch (event.type) {
    case "node_created":
      return [{ op: "node_added", nodeId: p.nodeId as string, data: p }];

    case "node_deleted": {
      const nodeIds = p.nodeIds as string[] | undefined;
      const edgeIds = p.removedEdgeIds as string[] | undefined;
      const ops: DeltaOperation[] = [];
      if (edgeIds && edgeIds.length > 0) {
        for (const eid of edgeIds) {
          ops.push({ op: "edge_removed", edgeId: eid });
        }
      }
      if (nodeIds && nodeIds.length > 0) {
        for (const nid of nodeIds) {
          ops.push({ op: "node_removed", nodeId: nid });
        }
      }
      return ops;
    }

    case "node_updated": {
      const changes = (p.changes ?? {}) as Record<string, unknown>;
      return [{ op: "node_updated", nodeId: p.nodeId as string, changes }];
    }

    case "node_moved": {
      const position = p.position as { x: number; y: number };
      return [{ op: "node_moved", nodeId: p.nodeId as string, position }];
    }

    case "edge_created":
      return [
        { op: "edge_added", edgeId: p.edgeId as string, data: p },
      ];

    case "edge_deleted": {
      const edgeIds = p.edgeIds as string[] | undefined;
      if (!edgeIds) return [];
      return edgeIds.map((eid: string) => ({
        op: "edge_removed" as const,
        edgeId: eid,
      }));
    }

    case "edge_reconnected":
      return [
        {
          op: "edge_reconnected",
          edgeId: p.edgeId as string,
          changes: p,
        },
      ];

    case "selection_changed":
      return [
        {
          op: "selection_changed",
          nodeIds: (p.nodeIds as string[]) ?? [],
          edgeIds: (p.edgeIds as string[]) ?? [],
        },
      ];

    case "graph_changed":
      // Graph-level change is implicit in the other events
      return [];

    case "subflow_toggled":
      return [
        {
          op: "node_updated",
          nodeId: p.nodeId as string,
          changes: { hidden: Boolean(p.collapsed) },
        },
      ];

    case "diagram_reset":
    case "diagram_imported":
      return [
        {
          op: "graph_reset",
          nodes: (p.nodes ?? []) as Node[],
          edges: (p.edges ?? []) as Edge[],
        },
      ];

    default:
      return [];
  }
}
