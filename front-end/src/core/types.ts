/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
