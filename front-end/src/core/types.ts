// Re-export Svelte Flow types (type-only — no runtime dependency)
import type { Node, Edge } from "@xyflow/svelte";
export type { Node, Edge };

// ── Domain Events (EventBus) ──────────────────
export type DomainEventType =
  | "node_created"
  | "node_deleted"
  | "node_updated"
  | "node_moved"
  | "edge_created"
  | "edge_deleted"
  | "edge_reconnected"
  | "subflow_toggled"
  | "selection_changed"
  | "graph_changed"
  | "diagram_reset"
  | "diagram_imported";

export interface DomainEvent<T = Record<string, unknown>> {
  type: DomainEventType;
  seq: number;
  timestamp: number;
  transactionId?: string;
  payload: T;
}

// ── Position ────────────────────────────────────
export interface Position { x: number; y: number; }

// ── Node Configuration ──────────────────────────
export interface NodeConfig {
  name?: string;
  color?: string;
  width?: number;
  height?: number;
  params?: Record<string, any>;
}

export interface JoinNodeConfig extends NodeConfig {
  inputsCount?: number;
}

// ── Snapshots ───────────────────────────────────
export interface DiagramCoreSnapshot {
  nodes: Node[];
  edges: Edge[];
}
