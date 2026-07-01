import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [
    svelte({
      emitCss: false,
    }),
  ],
  server: {
    proxy: {
      "/ws": {
        target: "ws://localhost:9339",
        ws: true,
      },
    },
  },
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
});
