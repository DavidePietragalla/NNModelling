import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [
    svelte({
      emitCss: false,
    }),
  ],
  test: {
    include: ["src/__tests__/**/*.test.ts"],
    exclude: ["src/__tests__/integration/**"],
    globals: true,
  },
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
});
