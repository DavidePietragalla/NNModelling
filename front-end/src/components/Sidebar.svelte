<script lang="ts">
  import SDropdown from "./SDropdown.svelte";
  import type { Diagram } from "../Diagram.svelte";
  import type { Stereotype } from "../stereotype";
  import type { Node } from "@xyflow/svelte";
  import { TypeEngine } from "../conversion/typeEngine";

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

  // --- TYPE CHECK - Debounced inference on param change ---
  let typeCheckTimer: ReturnType<typeof setTimeout>;

  function scheduleTypeCheck() {
    clearTimeout(typeCheckTimer);
    typeCheckTimer = setTimeout(() => {
      diagram.typeResult = TypeEngine.infer(diagram);
    }, 300);
  }

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
    scheduleTypeCheck();
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
    scheduleTypeCheck();
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
    scheduleTypeCheck();
  }

  function getNodeLabel(nodeId: string): string {
    const n = diagram.nodes.find(node => node.id === nodeId);
    if (!n) return nodeId;
    return (n.data as Record<string, unknown>)?.name as string ?? (n.data as Record<string, unknown>)?.stereotype as string ?? nodeId;
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
      scheduleTypeCheck();
    }
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
        <input type="text" bind:value={form.name} oninput={handleManualUpdate} />
      </label>

      <div class="row">
        {#if !isSubflow || (isSubflow && selection !== null)}
          <label>Colore <input type="color" bind:value={form.color} oninput={handleManualUpdate} /></label>
        {/if}
        <label>Width <input type="number" bind:value={form.width} oninput={handleManualUpdate} /></label>
        <label>Height <input type="number" bind:value={form.height} oninput={handleManualUpdate} /></label>
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
                <input type="text" id={key} bind:value={form.params[key].value} oninput={handleManualUpdate} />
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
                <input type="text" id={key} bind:value={form.params[key].value} oninput={handleManualUpdate} />
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

    {#if diagram.typeResult}
      <div class="type-error-panel">
        <div class="type-error-panel-header">
          Type Check ({diagram.typeResult.errors.length} issues)
        </div>
        {#if diagram.typeResult.errors.length === 0}
          <div class="type-errors-empty">No type errors or warnings.</div>
        {:else}
          {#each diagram.typeResult.errors as err}
            <!-- svelte-ignore a11y_no_static_element_interactions a11y_click_events_have_key_events -->
            <div class="type-error-item {err.severity}" onclick={() => {
              const found = diagram.nodes.find(n => n.id === err.nodeId);
              if (found) {
                diagram.nodes = diagram.nodes.map(n => ({ ...n, selected: n.id === found.id }));
              }
            }} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { const found = diagram.nodes.find(n => n.id === err.nodeId); if (found) { diagram.nodes = diagram.nodes.map(n => ({ ...n, selected: n.id === found.id })); } } }} role="button" tabindex="0">
              <span class="type-error-icon">{err.severity === 'error' ? '❌' : '⚠️'}</span>
              <div class="type-error-text">
                <span class="type-error-node">{getNodeLabel(err.nodeId)}</span>
                <span class="type-error-msg">{err.message}</span>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    {/if}

  </aside>
{/if}

<style>
  @import "../styles/sidebar.css";

  .type-error-panel {
    margin-top: 16px;
    border-top: 1px solid #e5e7eb;
    padding-top: 8px;
    padding-left: 8px;
    padding-right: 8px;
  }
  .type-error-panel-header {
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 6px;
  }
  .type-errors-empty {
    font-style: italic;
    color: #6b7280;
    font-size: 0.8rem;
  }
  .type-error-item {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    padding: 3px 0;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .type-error-item:hover {
    background: #f3f4f6;
  }
  .type-error-item.error .type-error-msg {
    color: #dc2626;
  }
  .type-error-item.warning .type-error-msg {
    color: #f59e0b;
  }
  .type-error-icon {
    flex-shrink: 0;
  }
  .type-error-text {
    display: flex;
    flex-direction: column;
  }
  .type-error-node {
    font-weight: 600;
  }
</style>
