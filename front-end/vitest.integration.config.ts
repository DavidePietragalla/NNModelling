import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

/**
 * NNModelling integration test configuration.
 *
 * Tiers are selected via NNM_TIER (smoke | convert | forward | train | infer |
 * all) and mirrored in front-end/package.json scripts. A tier is explicitly
 * *excluded* when NNM_TIER does not match; tests that require the Python
 * pipeline skip when the runtime is missing, unless NNM_REQUIRE_PYTHON=true
 * (CI-required mode), in which case a missing runtime fails the run instead of
 * silently skipping.
 */
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
      NNM_DEVICE_COUNT: process.env.NNM_DEVICE_COUNT || "1",
      NNM_DIAGRAM: process.env.NNM_DIAGRAM || "",
      NNM_TIER: process.env.NNM_TIER || "all",
      NNM_WANDB_MODE: process.env.NNM_WANDB_MODE || "disabled",
      NNM_REQUIRE_PYTHON: process.env.NNM_REQUIRE_PYTHON || "false",
    },
  },
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
});
