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
  import SDropdown from "./SDropdown.svelte";
  import type { Diagram } from "../Diagram.svelte";
  import type { Stereotype } from "../stereotype";
  import type { Node } from "@xyflow/svelte";

  interface Props {
    diagram: Diagram;
    selectedNode: Node | null;
    isOpen: boolean;
    onClose: () => void;
    getSpawnPosition: () => {x: number, y: number};
  }

  let { diagram, selectedNode, isOpen, onClose, getSpawnPosition }: Props = $props();

  // --- STATO DEL FORM ---
  let form = $state({
    name: "",
    color: "#4779c4",
    width: 140,
    height: 60,
    params: {} as Record<string, any>,
  });

  let selection = $state<Stereotype | null>(null);
  let isEditing = $derived(selectedNode !== null);
  let isSubflow = $derived(selectedNode?.type === "subflow"); // <-- Rileva se è un sottografo
  let subflowStereotypes = $derived(diagram.stereotypes.filter(s => s.isSubFlow));

  // --- LOGICA RESIZER ---
  let sidebarWidth = $state(320);
  let isDragging = $state(false);

  function startResize(e: MouseEvent) {
    isDragging = true;
    window.addEventListener('mousemove', doResize);
    window.addEventListener('mouseup', stopResize);
  }

  function doResize(e: MouseEvent) {
    if (isDragging) {
      const newWidth = window.innerWidth - e.clientX;
      if (newWidth > 200 && newWidth < 600) {
        sidebarWidth = newWidth;
      }
    }
  }

  function stopResize() {
    isDragging = false;
    window.removeEventListener('mousemove', doResize);
    window.removeEventListener('mouseup', stopResize);
  }

  let lastLoadedId = $state<string | null>(null);

  $effect(() => {
    const id = selectedNode?.id ?? null;
    if (id === lastLoadedId) return;
    lastLoadedId = id;
    if (selectedNode) {
      loadExistingNode(selectedNode);
    } else {
      resetForm();
    }
  });

  function loadExistingNode(vnode: Node) {
    if (vnode.type === "subflow") {
      const stereotypeName = vnode.data.stereotype as string;
      selection = diagram.stereotypes.find((s) => s.name === stereotypeName) || null;
      form.name = (vnode.data.name as string) || (vnode.data.label as string) || "";
      form.color = (vnode.data.color as string) || "#9b59b6";
      form.width = vnode.width || 400;
      form.height = vnode.height || 300;
      form.params = {};
    } else {
      // Gestione standard per nodi Custom
      const stereotypeName = vnode.data.stereotype as string;
      selection = diagram.stereotypes.find((s) => s.name === stereotypeName) || null;
      form.name = (vnode.data.name as string) || "";
      form.color = (vnode.data.color as string) || "#4779c4";
      form.width = vnode.width || 140;
      form.height = vnode.height || 60;
      form.params = {};
    }

    // Build params from stereotype defaults, then override with stored values
    if (selection) {
      let merged: Record<string, any> = {};
      for (const [key, prop] of Object.entries(selection.parameters || {})) {
        merged[key] = { value: prop.default, position: prop.position };
      }
      for (const [key, val] of Object.entries(vnode.data.params || {})) {
        merged[key] = { ...merged[key], ...JSON.parse(JSON.stringify(val)) };
      }
      form.params = merged;
    }
  }

  function resetForm() {
    selection = null;
    form.name = "";
    form.color = "#4779c4";
    form.width = 140;
    form.height = 60;
    form.params = {};
  }

  function onStereotypeChange(newStereotype: Stereotype | null) {
    selection = newStereotype;
    if (!newStereotype) {
      form.params = {};
      return;
    }
    form.color = newStereotype.view?.color ?? "#4779c4";
    form.width = newStereotype.view?.width ?? 140;
    form.height = newStereotype.view?.height ?? 60;

    let initialValues: Record<string, any> = {};
    for (const [key, prop] of Object.entries(newStereotype.parameters || {})) {
      initialValues[key] = {
        value: prop.default,
        position: prop.position
      }
    }
    form.params = initialValues;
  }

  function onSubflowStereotypeChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    const found = diagram.stereotypes.find(s => s.name === target.value && s.isSubFlow) || null;
    selection = found;
    if (!found) {
      form.params = {};
      return;
    }
    form.color = found.view?.color ?? "#9b59b6";
    form.width = found.view?.width ?? 400;
    form.height = found.view?.height ?? 300;
    let initialValues: Record<string, any> = {};
    for (const [key, prop] of Object.entries(found.parameters || {})) {
      initialValues[key] = { value: prop.default, position: prop.position };
    }
    form.params = initialValues;
  }

  function handleCreate() {
    if (!selection) return alert("Scegli uno stereotipo!");
 
    const pos = getSpawnPosition();

    if (!selection.isJoin) {
      diagram.addModule(selection, pos.x, pos.y, {
        name: form.name,
        color: form.color,
        width: form.width,
        height: form.height,
        params: { ...form.params }
      });
    } else {
      diagram.addJoinNode(selection, pos.x, pos.y, {
        name: form.name,
        color: form.color,
        params: { ...form.params }
      });
    }
    resetForm();
  }

  function handleManualUpdate() {
    if (isEditing && selectedNode) {
      // Configuriamo il payload in base al tipo di nodo
      if (isSubflow) {
        const updateConfig: any = {
          label: form.name,
          width: form.width,
          height: form.height
        };
        if (selection) {
          updateConfig.name = form.name;
          updateConfig.color = form.color;
          updateConfig.stereotype = selection.name;
          updateConfig.params = { ...form.params };
        }
        diagram.updateModule(selectedNode.id, updateConfig);
      } else {
        diagram.updateModule(selectedNode.id, {
          name: form.name,
          color: form.color,
          width: form.width,
          height: form.height,
          stereotype: selection?.name,
          params: { ...form.params }
        });
      }
    }
  }

 function handleLiveUpdate() {
    handleManualUpdate();
  } 
</script>

{#if isOpen}
  <aside class="sidebar" style="width: {sidebarWidth}px; user-select: {isDragging ? 'none' : 'auto'};">
    
    <div class="resizer" onmousedown={startResize} role="separator" aria-orientation="vertical" tabindex="0"></div>

    <div class="sidebar-header">
      <h3>
        {#if !isEditing}
          Nuovo Nodo
        {:else if isSubflow}
          Modifica Subflow
        {:else}
          Modifica Nodo
        {/if}
      </h3>
      <button class="close-btn" onclick={onClose}>✖</button>
    </div>

    <div class="form-container">
      <label>
        {isSubflow ? "Etichetta Sottografo" : "Nome"}
        <input type="text" bind:value={form.name} oninput={handleLiveUpdate} />
      </label>

      <div class="row">
        {#if !isSubflow || (isSubflow && selection !== null)}
          <label>Colore <input type="color" bind:value={form.color} oninput={handleLiveUpdate} /></label>
        {/if}
        <label>Width <input type="number" bind:value={form.width} oninput={handleLiveUpdate} /></label>
        <label>Height <input type="number" bind:value={form.height} oninput={handleLiveUpdate} /></label>
      </div>

      {#if isSubflow}
        <div>
          <label>Stereotipo SubFlow</label>
          <select onchange={onSubflowStereotypeChange}>
            <option value="">-- nessuno stereotipo --</option>
            {#each subflowStereotypes as stype}
              <option value={stype.name} selected={selection?.name === stype.name}>{stype.name}</option>
            {/each}
          </select>
        </div>

        {#if selection !== null}
          <div class="params-section">
            <h4>Parametri</h4>
            {#each Object.entries(selection.parameters || {}) as [key, config]}
              <div class="param-row">
                <label for={key}>{key}</label>
                <input type="text" id={key} bind:value={form.params[key].value} oninput={handleLiveUpdate} />
              </div>
            {/each}
          </div>
        {/if}
      {/if}

      {#if !isSubflow}
        <div>
          <label>Stereotipo</label>
          <SDropdown {diagram} selectedStereotype={selection} onSelectedChange={onStereotypeChange} />
        </div>

        {#if selection !== null}
          <div class="params-section">
            <h4>Parametri</h4>
            {#each Object.entries(selection.parameters || {}) as [key, config]}
              <div class="param-row">
                <label for={key}>{key}</label>
                <input type="text" id={key} bind:value={form.params[key].value} oninput={handleLiveUpdate} />
              </div>
            {/each}
          </div>
        {/if}
      {/if}

      {#if !isEditing}
        {#if selection !== null}
          <button class="create-btn" onclick={handleCreate}>➕ Aggiungi al Canvas</button>
        {/if}
      {:else}
        <button class="update-btn" onclick={handleManualUpdate}>💾 Salva Modifiche</button>
      {/if}
      
    </div>
  </aside>
{/if}

<style>
  @import "../styles/sidebar.css";
</style>
