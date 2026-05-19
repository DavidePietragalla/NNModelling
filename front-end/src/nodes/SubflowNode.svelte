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
</script>

<NodeResizer minWidth={250} minHeight={250} isVisible={selected} />

<Handle type="target" position={Position.Top} />

<div class="subflow-wrapper">
  <div class="subflow-header" style:background={String(data.color || "#007bff")}>
    {data.name || data.label}
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

  <div class="subflow-body"></div>

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

  .subflow-wrapper {
    padding-top: 0;
  }

  .subflow-header {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    color: white;
    font-weight: bold;
    font-size: 0.85rem;
    border-radius: 6px 6px 0 0;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .subflow-body {
    flex-grow: 1;
  }
</style>
