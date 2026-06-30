<script lang="ts">
  import {
    Handle,
    Position,
    NodeResizer,
    type NodeProps,
  } from "@xyflow/svelte";
  import { getContext } from "svelte";
  import { DIAGRAM_CONTEXT_KEY, type Diagram } from "../Diagram.svelte";

  let { data, selected, isConnectable, id }: NodeProps = $props();
  const diagram = getContext<Diagram>(DIAGRAM_CONTEXT_KEY);

  // Svelte 5: Filtriamo dinamicamente i parametri per posizione
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

  function focusInSidebar() {
    diagram.nodes = diagram.nodes.map((n) => ({
      ...n,
      selected: n.id === id,
    }));
  }
</script>

<NodeResizer
  minWidth={140}
  minHeight={80}
  isVisible={selected && !data.isInput}
/>

{#if !data.isInput}
  <Handle type="target" position={Position.Top} {isConnectable} />
{/if}

{#if data.isInput}
  <div class="input-circle" style="position: relative;">
    <div class="input-label">{data.name}</div>
  </div>
{:else}
  <div
    class="node-body"
    style="background-color: {(data.color as string) || 'white'};
           color: {(data.color as string) ? 'white' : 'black'};
           text-shadow: {(data.color as string) ? '1px 1px 2px rgba(0,0,0,0.8)' : 'none'};
           position: relative;"
  >
    <div class="params-container top-params">
      {#each topParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>

    <div class="node-title">
      {data.name || "Senza Nome"}
    </div>

    <div class="params-container bottom-params">
      {#each bottomParams as [key, param]}
        <div class="param-row">
          <span class="param-key">{key}</span>
          <span class="param-value">{param.value}</span>
        </div>
      {/each}
    </div>
  </div>
{/if}

{#if !data.isLoss}
  <Handle type="source" position={Position.Bottom} {isConnectable} />
{/if}

<style>
  @import "../styles/node.css";
</style>
