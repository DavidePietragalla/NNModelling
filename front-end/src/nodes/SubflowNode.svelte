<script lang="ts">
  import { Handle, Position, NodeResizer, type NodeProps } from "@xyflow/svelte";

  let { data, selected }: NodeProps = $props();

  let topParams = $derived(
    Object.entries(data.params || {}).filter(
      ([_, p]: any) => p?.position === "top",
    ),
  );

  let bottomParams = $derived(
    Object.entries(data.params || {}).filter(
      ([_, p]: any) => p?.position === "bottom",
    ),
  );

  let borderColor = $derived(String(data.color || "#007bff"));
  let bgColor = $derived(data.color ? `${String(data.color)}11` : "rgba(0, 123, 255, 0.05)");
</script>

<NodeResizer minWidth={250} minHeight={250} isVisible={selected} />

<Handle type="target" position={Position.Top} />

<div
  class="subflow-wrapper"
  style:border-color={borderColor}
  style:background={bgColor}
>
  <div class="subflow-label" style:background={borderColor}>
    {data.stereotype || data.label}
  </div>

  {#if data.stereotype && topParams.length > 0}
    <div class="params-container top-params">
      {#each topParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}

  <div class="subflow-name">
    {#if data.stereotype}
      {data.name || data.label}
    {:else}
      {data.label}
    {/if}
  </div>

  {#if data.stereotype && bottomParams.length > 0}
    <div class="params-container bottom-params">
      {#each bottomParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<Handle type="source" position={Position.Bottom} />

<style>
  @import "../styles/subflow.css";
</style>
