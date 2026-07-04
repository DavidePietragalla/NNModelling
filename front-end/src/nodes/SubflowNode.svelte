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
  import { type Node } from "@xyflow/svelte";
  import { getContext } from "svelte";
  import type { DiagramCore } from "../core/DiagramCore";

  const diagram = getContext<DiagramCore>("diagram");

  type SubflowData = {
    label: string;
    isCollapsed: boolean;
    color: any;
    params: Object;
    stereotype: any;
  };

  type MySubflowNode = Node<SubflowData, "subflow">;

  let { data, selected, id }: NodeProps<MySubflowNode> = $props();

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
  minWidth={200}
  minHeight={50}
  isVisible={selected}
/>

<Handle type="target" position={Position.Top} />

<div class="subflow-wrapper" class:collapsed={data.isCollapsed}>
  <div
    class="subflow-header"
    style:background={String(data.color || "#007bff")}
  >
    {data.label || ""}
    <button
      class="collapse-btn"
      onclick={() => diagram.toggleSubflow(id, !data.isCollapsed)}
    >
      {data.isCollapsed ? "+" : "-"}
    </button>
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
</style>
