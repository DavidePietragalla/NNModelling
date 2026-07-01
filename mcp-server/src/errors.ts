/**
 * Error type hierarchy for the NNModelling MCP server.
 *
 * All error classes extend the base `MCPServerError` class, which provides
 * a machine-readable `code`, a human-readable `message`, and optional `details`.
 */

// ── Base Error ──────────────────────────────────

export class MCPServerError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "MCPServerError";
  }

  toJSON() {
    return {
      code: this.code,
      message: this.message,
      details: this.details,
    };
  }
}

// ── Validation Errors ──────────────────────────

export class StereotypeNotFoundError extends MCPServerError {
  constructor(name: string) {
    super("STEREOTYPE_NOT_FOUND", `Stereotype '${name}' not found`, { stereotypeName: name });
  }
}

export class NodeNotFoundError extends MCPServerError {
  constructor(nodeId: string) {
    super("NODE_NOT_FOUND", `Node '${nodeId}' not found`, { nodeId });
  }
}

export class EdgeNotFoundError extends MCPServerError {
  constructor(edgeId: string) {
    super("EDGE_NOT_FOUND", `Edge '${edgeId}' not found`, { edgeId });
  }
}

export class ParameterNotFoundError extends MCPServerError {
  constructor(nodeId: string, key: string) {
    super("PARAMETER_NOT_FOUND", `Parameter '${key}' not found on node '${nodeId}'`, { nodeId, key });
  }
}

export class ParameterTypeMismatchError extends MCPServerError {
  constructor(nodeId: string, key: string, expected: string, received: string) {
    super(
      "PARAMETER_TYPE_MISMATCH",
      `Parameter '${key}' on node '${nodeId}' expected type '${expected}', got '${received}'`,
      { nodeId, key, expected, received }
    );
  }
}

export class TargetHandleOccupiedError extends MCPServerError {
  constructor(target: string, targetHandle: string) {
    super(
      "TARGET_HANDLE_OCCUPIED",
      `Target handle '${targetHandle}' on node '${target}' is already connected`,
      { target, targetHandle }
    );
  }
}

export class InvalidConnectionError extends MCPServerError {
  constructor(reason: string) {
    super("INVALID_CONNECTION", reason);
  }
}

export class SelfLoopError extends MCPServerError {
  constructor(nodeId: string) {
    super("SELF_LOOP", `Cannot connect node '${nodeId}' to itself`, { nodeId });
  }
}

export class CycleDetectedError extends MCPServerError {
  constructor(nodeId: string) {
    super("CYCLE_DETECTED", `Connection would create a cycle involving node '${nodeId}'`, { nodeId });
  }
}

export class InvalidPositionError extends MCPServerError {
  constructor(x: unknown, y: unknown) {
    super("INVALID_POSITION", `Invalid position: (${x}, ${y})`, { x, y });
  }
}

export class InvalidSubflowError extends MCPServerError {
  constructor(nodeId: string, reason: string) {
    super("INVALID_SUBFLOW", `Invalid subflow '${nodeId}': ${reason}`, { nodeId, reason });
  }
}

// ── Pipeline Errors ────────────────────────────

export class CompilationFailedError extends MCPServerError {
  constructor(reason: string) {
    super("COMPILATION_FAILED", reason);
  }
}

export class ConversionFailedError extends MCPServerError {
  constructor(reason: string) {
    super("CONVERSION_FAILED", reason);
  }
}

export class TrainingFailedError extends MCPServerError {
  constructor(reason: string) {
    super("TRAINING_FAILED", reason);
  }
}

export class InferenceFailedError extends MCPServerError {
  constructor(reason: string) {
    super("INFERENCE_FAILED", reason);
  }
}

// ── Transaction Errors ─────────────────────────

export class NoActiveTransactionError extends MCPServerError {
  constructor() {
    super("NO_ACTIVE_TRANSACTION", "No active transaction");
  }
}

export class TransactionAlreadyActiveError extends MCPServerError {
  constructor() {
    super("TRANSACTION_ALREADY_ACTIVE", "A transaction is already active");
  }
}

// ── History Errors ─────────────────────────────

export class NothingToUndoError extends MCPServerError {
  constructor() {
    super("NOTHING_TO_UNDO", "Nothing to undo");
  }
}

export class NothingToRedoError extends MCPServerError {
  constructor() {
    super("NOTHING_TO_REDO", "Nothing to redo");
  }
}

// ── Serialization Errors ───────────────────────

export class ImportFailedError extends MCPServerError {
  constructor(reason: string) {
    super("IMPORT_FAILED", reason);
  }
}

export class ExportFailedError extends MCPServerError {
  constructor(reason: string) {
    super("EXPORT_FAILED", reason);
  }
}
