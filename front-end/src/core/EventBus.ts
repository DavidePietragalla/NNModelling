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

import type { DomainEvent, DomainEventType } from "./types";

export type EventHandler<T = Record<string, unknown>> = (event: DomainEvent<T>) => void;

export class EventBus {
  private handlers = new Map<DomainEventType, Set<EventHandler<any>>>();
  private anyHandlers = new Set<EventHandler<any>>();
  private seq: number = 0;
  private buffer: DomainEvent[] = [];
  private readonly maxBufferSize: number = 1000;

  /** Subscribe to a specific event type. Returns unsubscribe function. */
  on<T>(type: DomainEventType, handler: EventHandler<T>): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
    return () => this.handlers.get(type)?.delete(handler);
  }

  /** Subscribe to ALL event types. Used by WebSocket server for blind broadcast. */
  onAny(handler: EventHandler): () => void {
    this.anyHandlers.add(handler);
    return () => this.anyHandlers.delete(handler);
  }

  /** Emit an event. Called synchronously by DiagramCore after every mutation. */
  emit<T>(type: DomainEventType, payload: T, transactionId?: string): void {
    this.seq++;
    const event: DomainEvent<T> = {
      type,
      seq: this.seq,
      timestamp: Date.now(),
      transactionId,
      payload,
    };

    // Ring buffer for late subscribers
    this.buffer.push(event as DomainEvent);
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer.shift();
    }

    // Notify type-specific handlers
    const typed = this.handlers.get(type);
    if (typed) {
      for (const h of typed) h(event);
    }

    // Notify catch-all handlers (WebSocket server)
    for (const h of this.anyHandlers) h(event);
  }

  /** Get all events since a given sequence number (exclusive). Used by MCP get_events. */
  getEventsSince(lastSeq: number): DomainEvent[] {
    return this.buffer.filter(e => e.seq > lastSeq);
  }

  getCurrentSeq(): number { return this.seq; }

  clear(): void {
    this.seq = 0;
    this.buffer = [];
    // Handlers are NOT cleared — only the event log is reset
  }
}
