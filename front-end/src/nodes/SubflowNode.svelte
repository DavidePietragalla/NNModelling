<script lang="ts">
  import { NodeResizer, type NodeProps } from "@xyflow/svelte";

  let { data, selected }: NodeProps = $props();
</script>

<NodeResizer minWidth={250} minHeight={250} isVisible={selected} />

<div class="subflow-wrapper">
  <div class="subflow-label">
    {data.label}
  </div>
</div>

<style>
  @import "../styles/subflow.css";
</style>
