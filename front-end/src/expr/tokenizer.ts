/**
 * @file Tokenizer for the expression language.
 *
 * Converts a source string into a sequence of tokens.
 * Handles $-prefixed identifiers and $* as single tokens.
 */

import { ParseError } from "./types";
import type { Token, TokenKind } from "./types";

/**
 * Tokenize an expression source string.
 *
 * Grammar:
 *   NUMBER       → \d+
 *   IDENTIFIER   → [a-zA-Z_][a-zA-Z0-9_]*
 *   DOLLAR_IDENT → \$[a-zA-Z_][a-zA-Z0-9_]*
 *   DOLLAR_STAR  → \$\*
 */
export function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  // Sentinel: when we've consumed everything, the loop ends
  while (i < source.length) {
    const ch = source[i];

    // ── Whitespace ──────────────────────────────────────────
    if (ch === " " || ch === "\t" || ch === "\n" || ch === "\r") {
      i++;
      continue;
    }

    // ── Single-char tokens ──────────────────────────────────
    if (ch === "+") {
      tokens.push({ kind: "PLUS", value: "+", pos: i });
      i++;
      continue;
    }
    if (ch === "-") {
      tokens.push({ kind: "MINUS", value: "-", pos: i });
      i++;
      continue;
    }
    if (ch === "*") {
      // Check for $* — but that's handled below in the $ branch.
      // Bare * is not valid in the grammar, but we tokenize it anyway
      // and let the parser reject it.
      tokens.push({ kind: "STAR", value: "*", pos: i });
      i++;
      continue;
    }
    if (ch === "/") {
      // Check for //
      if (i + 1 < source.length && source[i + 1] === "/") {
        tokens.push({ kind: "FLOOR_DIV", value: "//", pos: i });
        i += 2;
      } else {
        tokens.push({ kind: "SLASH", value: "/", pos: i });
        i++;
      }
      continue;
    }
    if (ch === "%") {
      tokens.push({ kind: "PERCENT", value: "%", pos: i });
      i++;
      continue;
    }
    if (ch === "(") {
      tokens.push({ kind: "LPAREN", value: "(", pos: i });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ kind: "RPAREN", value: ")", pos: i });
      i++;
      continue;
    }
    if (ch === ",") {
      tokens.push({ kind: "COMMA", value: ",", pos: i });
      i++;
      continue;
    }

    // ── Dollar-prefixed tokens ──────────────────────────────
    if (ch === "$") {
      // Check for $* (DOLLAR_STAR)
      if (i + 1 < source.length && source[i + 1] === "*") {
        tokens.push({ kind: "DOLLAR_STAR", value: "$*", pos: i });
        i += 2;
        continue;
      }

      // $ followed by identifier → DOLLAR_IDENT
      if (i + 1 < source.length && isIdentStart(source[i + 1])) {
        const start = i;
        i++; // skip $
        while (i < source.length && isIdentChar(source[i])) {
          i++;
        }
        const name = source.slice(start + 1, i);
        tokens.push({ kind: "DOLLAR_IDENT", value: name, pos: start });
        continue;
      }

      // Lone $ → error
      throw new ParseError("Unexpected '$' — expected '$*' or '$IDENTIFIER'", i);
    }

    // ── Number ──────────────────────────────────────────────
    if (isDigit(ch)) {
      const start = i;
      while (i < source.length && isDigit(source[i])) {
        i++;
      }
      // Optional decimal part
      if (i < source.length && source[i] === ".") {
        i++;
        while (i < source.length && isDigit(source[i])) {
          i++;
        }
      }
      tokens.push({ kind: "NUMBER", value: source.slice(start, i), pos: start });
      continue;
    }

    // ── Identifier (without $) ──────────────────────────────
    if (isIdentStart(ch)) {
      const start = i;
      while (i < source.length && isIdentChar(source[i])) {
        i++;
      }
      tokens.push({
        kind: "IDENTIFIER",
        value: source.slice(start, i),
        pos: start,
      });
      continue;
    }

    // ── Unknown character ───────────────────────────────────
    throw new ParseError(`Unexpected character '${ch}'`, i);
  }

  return tokens;
}

function isDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9";
}

function isIdentStart(ch: string): boolean {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_";
}

function isIdentChar(ch: string): boolean {
  return isIdentStart(ch) || isDigit(ch);
}
