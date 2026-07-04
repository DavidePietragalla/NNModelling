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
  import { Handle, Position, useSvelteFlow, type NodeProps } from "@xyflow/svelte";

  let { id, data, selected, isConnectable }: NodeProps = $props();
  
  // Usiamo l'API nativa per aggiornare i dati del nodo senza impazzire con classi esterne
  const { updateNodeData } = useSvelteFlow();

  // Reattività nativa Svelte 5 basata sul payload "data"
  let inputsCount = $derived((data.inputsCount as number) || 2);
  let name = $derived((data.name as string) || "Join");

  function increase(e: Event) {
    e.stopPropagation();
    updateNodeData(id, { inputsCount: inputsCount + 1 });
  }

  function decrease(e: Event) {
    e.stopPropagation();
    if (inputsCount > 2) {
      updateNodeData(id, { inputsCount: inputsCount - 1 });
    }
  }
</script>

<div class="node-wrapper" class:selected>
  <button class="btn-branch" onclick={decrease} disabled={inputsCount <= 2}>
    -
  </button>

  <div class="join-center">
    {#each Array(inputsCount) as _, i}
      <Handle
        type="target"
        position={Position.Top}
        id={`in-${i}`}
        {isConnectable}
        style="left: {((i + 1) * 100) / (inputsCount + 1)}%;"
      />
    {/each}

    <div class="join-line" style="width: {inputsCount * 30}px;"></div>

    <Handle type="source" position={Position.Bottom} id="out" {isConnectable} />
  </div>

  <button class="btn-branch" onclick={increase}>+</button>

  <div class="join-label" title={name}>
    {name.length > 8 ? name.slice(0, 8) + '...' : name}
  </div>
</div>

<style>
  @import "../styles/join.css";
</style>
