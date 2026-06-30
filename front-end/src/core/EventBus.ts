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
