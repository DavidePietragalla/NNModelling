/**
 * @file Recursive descent parser for the expression language.
 *
 * Grammar:
 *   expr        := additive
 *   additive    := multiplicative (("+" | "-") multiplicative)*
 *   multiplicative := unary (("*" | "/" | "//" | "%") unary)*
 *   unary       := "-" unary | primary
 *   primary     := NUMBER | DOLLAR_IDENT | DOLLAR_STAR | IDENTIFIER
 *                | FUNC_CALL | "(" expr ")"
 *   FUNC_CALL   := IDENTIFIER "(" expr ("," expr)* ")"
 *
 * Precedence (lowest to highest):
 *   1. additive (+, -)
 *   2. multiplicative (*, /, //, %)
 *   3. unary (-)
 *   4. primary (numbers, variables, function calls, grouping)
 */

import { ParseError } from "./types";
import type { Token, ExprNode, BinaryOp } from "./types";

/**
 * Parse a token sequence into an AST.
 */
export function parse(tokens: Token[]): ExprNode {
  if (tokens.length === 0) {
    throw new ParseError("Empty expression", 0);
  }

  let pos = 0;

  function peek(): Token | undefined {
    return tokens[pos];
  }

  function consume(kind?: string): Token {
    const token = tokens[pos];
    if (!token) {
      throw new ParseError(
        `Unexpected end of expression${kind ? `, expected ${kind}` : ""}`,
        pos > 0 ? tokens[pos - 1].pos + tokens[pos - 1].value.length : 0,
      );
    }
    if (kind !== undefined && token.kind !== kind) {
      throw new ParseError(
        `Expected ${kind} at position ${token.pos}, got '${token.value}'`,
        token.pos,
      );
    }
    pos++;
    return token;
  }

  // ── expr := additive ───────────────────────────────────────────
  function parseExpr(): ExprNode {
    return parseAdditive();
  }

  // ── additive := multiplicative (("+" | "-") multiplicative)* ──
  function parseAdditive(): ExprNode {
    let left = parseMultiplicative();
    while (pos < tokens.length) {
      const token = tokens[pos];
      if (token.kind === "PLUS") {
        pos++;
        const right = parseMultiplicative();
        left = { kind: "binary", op: "+", left, right };
      } else if (token.kind === "MINUS") {
        pos++;
        const right = parseMultiplicative();
        left = { kind: "binary", op: "-", left, right };
      } else {
        break;
      }
    }
    return left;
  }

  // ── multiplicative := unary (("*" | "/" | "//" | "%") unary)* ──
  function parseMultiplicative(): ExprNode {
    let left = parseUnary();
    while (pos < tokens.length) {
      const token = tokens[pos];
      if (
        token.kind === "STAR" ||
        token.kind === "SLASH" ||
        token.kind === "FLOOR_DIV" ||
        token.kind === "PERCENT"
      ) {
        pos++;
        const op = token.value as BinaryOp;
        const right = parseUnary();
        left = { kind: "binary", op, left, right };
      } else {
        break;
      }
    }
    return left;
  }

  // ── unary := "-" unary | primary ────────────────────────────────
  function parseUnary(): ExprNode {
    const token = tokens[pos];
    if (token && token.kind === "MINUS") {
      pos++;
      const operand = parseUnary();
      return { kind: "unary", op: "-", operand };
    }
    return parsePrimary();
  }

  // ── primary ────────────────────────────────────────────────────
  function parsePrimary(): ExprNode {
    const token = peek();
    if (!token) {
      throw new ParseError(
        "Unexpected end of expression",
        pos > 0 ? tokens[pos - 1].pos + tokens[pos - 1].value.length : 0,
      );
    }

    switch (token.kind) {
      case "NUMBER": {
        pos++;
        return { kind: "number", value: parseFloat(token.value) };
      }

      case "DOLLAR_IDENT": {
        pos++;
        return { kind: "variable", name: token.value, isSymbolic: true };
      }

      case "DOLLAR_STAR": {
        pos++;
        return { kind: "wildcard_product" };
      }

      case "IDENTIFIER": {
        // Could be a function call or a plain variable reference
        const name = token.value;
        pos++;

        // Peek ahead for '(' to detect function call
        if (pos < tokens.length && tokens[pos].kind === "LPAREN") {
          // Function call
          pos++; // consume '('
          const args: ExprNode[] = [];
          if (pos < tokens.length && tokens[pos].kind !== "RPAREN") {
            // Parse first argument
            args.push(parseExpr());
            // Parse remaining arguments
            while (pos < tokens.length && tokens[pos].kind === "COMMA") {
              pos++; // consume ','
              if (
                pos >= tokens.length ||
                tokens[pos].kind === "RPAREN"
              ) {
                throw new ParseError(
                  "Expected expression after ','",
                  pos > 0 ? tokens[pos - 1].pos + 1 : 0,
                );
              }
              args.push(parseExpr());
            }
          }
          if (pos >= tokens.length || tokens[pos].kind !== "RPAREN") {
            throw new ParseError(
              `Expected ')' after function call arguments`,
              pos > 0 ? tokens[pos - 1].pos + tokens[pos - 1].value.length : 0,
            );
          }
          pos++; // consume ')'
          return { kind: "call", name, args };
        }

        // Plain variable reference
        return { kind: "variable", name, isSymbolic: false };
      }

      case "LPAREN": {
        pos++; // consume '('
        const expr = parseExpr();
        if (pos >= tokens.length || tokens[pos].kind !== "RPAREN") {
          throw new ParseError(
            "Expected ')'",
            pos > 0 ? tokens[pos - 1].pos + tokens[pos - 1].value.length : 0,
          );
        }
        pos++; // consume ')'
        return expr;
      }

      default:
        throw new ParseError(
          `Unexpected token '${token.value}' at position ${token.pos}`,
          token.pos,
        );
    }
  }

  const result = parseExpr();

  // Ensure we consumed all tokens
  if (pos < tokens.length) {
    const remaining = tokens[pos];
    throw new ParseError(
      `Unexpected token '${remaining.value}' after expression`,
      remaining.pos,
    );
  }

  return result;
}
