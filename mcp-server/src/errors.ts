/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

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
}

// ── Pipeline Errors ────────────────────────────

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
