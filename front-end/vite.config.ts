import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  // GitHub Pages serves project sites below /<repository>/; locally Vite stays
  // available from the root URL.
  base: process.env.VITE_BASE_PATH ?? "/",
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
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  optimizeDeps: {
    exclude: ["@xyflow/svelte"],
  },
});
