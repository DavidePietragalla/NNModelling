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
