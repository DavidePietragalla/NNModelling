<!--
NNModelling — DSL for designing neural networks via visual node editor
Copyright (C) 2026  Luca Sforza

Licensed under the GNU General Public License v3 or later.
Commercial licenses are available — contact Luca Sforza.
See the LICENSE file for details.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
-->

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
