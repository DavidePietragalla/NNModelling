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

// ── WebSocket Messages ─────────────────────────
export type WSMessageType = "snapshot" | "delta";

export interface WSSnapshotMessage {
  type: "snapshot";
  seq: number;
  nodes: Node[];
  edges: Edge[];
}

export interface WSDeltaMessage {
  type: "delta";
  seq: number;
  operations: DeltaOperation[];
}

export type DeltaOperation =
  | { op: "node_added";    nodeId: string; data: Partial<Node> }
  | { op: "node_removed";  nodeId: string }
  | { op: "node_moved";    nodeId: string; position: { x: number; y: number } }
  | { op: "node_updated";  nodeId: string; changes: Record<string, unknown> }
  | { op: "edge_added";    edgeId: string; data: Partial<Edge> }
  | { op: "edge_removed";  edgeId: string }
  | { op: "edge_reconnected"; edgeId: string; changes: Record<string, unknown> }
  | { op: "selection_changed"; nodeIds: string[]; edgeIds: string[] }
  | { op: "graph_reset";   nodes: Node[]; edges: Edge[] };

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

export interface EdgeConfig {
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

// ── Selection ───────────────────────────────────
export interface Selection {
  nodeIds: string[];
  edgeIds: string[];
}

// ── Snapshots ───────────────────────────────────
export interface DiagramCoreSnapshot {
  nodes: Node[];
  edges: Edge[];
}

export interface DiagramSnapshot extends DiagramCoreSnapshot {
  timestamp: number;
  description: string;
}

// ── Canvas State ────────────────────────────────
export interface CanvasState {
  zoom: number;
  x: number;
  y: number;
}

// ── Graph Statistics ────────────────────────────
export interface GraphStatistics {
  nodeCount: number;
  edgeCount: number;
  moduleCount: number;
  joinCount: number;
  subflowCount: number;
  inputCount: number;
  lossCount: number;
  maxDepth: number;
  avgFanOut: number;
  cycleFree: boolean;
}

// ── NNTree Output ───────────────────────────────
export interface NNTreeOutput {
  json: string;
  root: string;
  nodeCount: number;
  subflowCount: number;
  lossNodeType: string | null;
}
