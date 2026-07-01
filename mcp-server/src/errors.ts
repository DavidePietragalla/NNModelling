/**
 * Error type hierarchy for the NNModelling MCP server.
 *
 * Only pipeline errors and the base class remain. Validation errors
 * (stereotype not found, invalid connection, etc.) are now handled by
 * the BrowserRPCHandler in the browser and propagated as plain Error
 * messages via the RPC response.
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
