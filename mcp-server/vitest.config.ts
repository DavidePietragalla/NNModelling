import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    include: ["__tests__/**/*.test.ts"],
    globals: true,
    testTimeout: 30_000, // 30s for tools that spawn subprocesses
  },
  resolve: {
    alias: {
      // @xyflow/svelte depends on Svelte runtime and is incompatible with
      // Node.js ESM resolution. Since all imports from @xyflow/svelte in
      // NNModelling core are type-only (Node, Edge interfaces), we redirect
      // to a minimal mock that satisfies Vite's module resolver.
      "@xyflow/svelte": path.resolve(__dirname, "__mocks__/@xyflow/svelte.ts"),
    },
  },
});
