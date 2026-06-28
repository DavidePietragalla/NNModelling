import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [
    svelte({ emitCss: false }),
  ],
  test: {
    include: ["src/__tests__/integration/**/*.test.ts"],
    globals: true,
    testTimeout: 600_000,      // 10 minutes max per test
    hookTimeout: 120_000,      // 2 minutes for setup/teardown
    pool: "forks",
    sequence: {
      concurrent: false,
    },
    env: {
      NNM_DEVICE: process.env.NNM_DEVICE || "cpu",
      NNM_DIAGRAM: process.env.NNM_DIAGRAM || "",
      NNM_TIER: process.env.NNM_TIER || "all",
      NNM_WANDB_MODE: process.env.NNM_WANDB_MODE || "disabled",
    },
  },
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
});
