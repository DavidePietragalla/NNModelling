/**
 * @file Type definitions for the expression evaluator.
 *
 * The expression language is a small arithmetic language used to declare
 * computed tensor dimensions in stereotype type signatures.
 */

import type { ShapeDimension } from "../conversion/tensortypes";

// ─────────────────────────────────────────────────────────────────────────────
// Tokens
// ─────────────────────────────────────────────────────────────────────────────

export type TokenKind =
  | "NUMBER"
  | "IDENTIFIER"
  | "DOLLAR_IDENT"
  | "DOLLAR_STAR"
  | "PLUS"
  | "MINUS"
  | "STAR"
  | "SLASH"
  | "FLOOR_DIV"
  | "PERCENT"
  | "LPAREN"
  | "RPAREN"
  | "COMMA";

export interface Token {
  kind: TokenKind;
  value: string;
  pos: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// AST
// ─────────────────────────────────────────────────────────────────────────────

export type BinaryOp = "+" | "-" | "*" | "/" | "//" | "%";

export type ExprNode =
  | { kind: "number"; value: number }
  | { kind: "variable"; name: string; isSymbolic: boolean }
  | { kind: "wildcard_product" }
  | { kind: "binary"; op: BinaryOp; left: ExprNode; right: ExprNode }
  | { kind: "unary"; op: "-"; operand: ExprNode }
  | { kind: "call"; name: string; args: ExprNode[] };

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation context
// ─────────────────────────────────────────────────────────────────────────────

export interface EvalContext {
  /** Symbolic environment from pattern matching (B, H, W, ...). */
  env: Map<string, ShapeDimension>;
  /** Wildcard-captured dimensions from pattern matching. */
  captured: ShapeDimension[];
  /** Node parameters to resolve bare identifiers. */
  params: Record<string, unknown>;
  /** Optional: resolve a parameter name to its numeric value.
   *  If not provided, falls back to reading from `params` directly. */
  resolveParam?: (name: string) => number | undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

export class ParseError extends Error {
  constructor(
    message: string,
    public position: number,
  ) {
    super(message);
    this.name = "ParseError";
  }
}
