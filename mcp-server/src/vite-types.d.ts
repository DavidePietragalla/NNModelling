/**
 * Type declarations for Vite-specific APIs used by @nnmodelling/front-end/core.
 *
 * The MCP server compiles with plain tsc (not Vite), so we need to declare
 * Vite's import.meta.glob here for type-checking to succeed when the server
 * imports DiagramCore → StereotypeCore.
 */

interface ImportMeta {
  glob<T = Record<string, unknown>>(
    pattern: string,
    options?: { eager?: boolean }
  ): Record<string, T>;
}
