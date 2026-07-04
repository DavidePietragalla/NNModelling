<!--
NNModelling — DSL for designing neural networks via visual node editor
Copyright (C) 2026  Luca Sforza

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

<script lang="ts">
  import {
    Handle,
    Position,
    NodeResizer,
    type NodeProps,
  } from "@xyflow/svelte";

  let { data, selected, isConnectable }: NodeProps = $props();

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
  <div class="input-circle">
    <div class="input-label">{data.name}</div>
  </div>
{:else}
  <div
    class="node-body"
    style:background-color={(data.color as string) || "white"}
    style:color={(data.color as string) ? "white" : "black"}
    style:text-shadow={(data.color as string) ? "1px 1px 2px rgba(0,0,0,0.8)" : "none"}
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

<Handle type="source" position={Position.Bottom} {isConnectable} />

<style>
  @import "../styles/node.css";
</style>
