<script lang="ts">
  import { SvelteFlowProvider } from "@xyflow/svelte";
  import FlowCanvas from "./FlowCanvas.svelte";
  import "@xyflow/svelte/dist/style.css";
</script>

<div
  style="height: 100vh; width: 100vw; overflow: hidden; background: #f8f8f8;"
>
  <SvelteFlowProvider>
    <FlowCanvas />
  </SvelteFlowProvider>
</div>
