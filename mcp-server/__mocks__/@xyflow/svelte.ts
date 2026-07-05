/**
 * Mock for @xyflow/svelte (Svelte Flow)
 *
 * The real package depends on Svelte runtime and is incompatible with
 * Node.js ESM resolution. Since all imports from @xyflow/svelte in the
 * NNModelling core are type-only (Node, Edge interfaces), we provide a
 * minimal mock that satisfies Vite's module resolution.
 *
 * All imports from this package are `import type`, so there is no runtime
 * dependency on this module. The exports here are purely to ensure Vite's
 * module resolver can find the package.
 */

// Runtime exports for Vite module resolution (not used, but must exist)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Node: any = {};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const Edge: any = {};
