<script lang="ts">
  import {
    SvelteFlow,
    MarkerType,
    Controls,
    Background,
    Panel,
    useSvelteFlow,
    type Connection,
    type Edge,
  } from "@xyflow/svelte";

  import Sidebar from "./components/Sidebar.svelte";

  const { getInternalNode, getIntersectingNodes, screenToFlowPosition } =
    useSvelteFlow();

  import CustomNode from "./nodes/CustomNode.svelte";
  import SubflowNode from "./nodes/SubflowNode.svelte";
  import JoinNode from "./nodes/JoinNode.svelte";
  import {
    checkValidConnection,
    handleLoadModel,
    handleSaveModel,
    onNodeDragStop,
  } from "./utils";

  import { NNTree } from "./conversion/nnTree";

  // 1. Importiamo la classe Diagram
  import { Diagram } from "./Diagram.svelte";
  import { toPng } from "html-to-image";

  // Context per SubflowNode — gli permette di chiamare diagram.toggleSubflow
  // senza bisogno di callback nel node data
  import { setContext } from "svelte";
  import type { DiagramCore } from "./core/DiagramCore";

  // RPC handler — receives MCP server requests and dispatches to Diagram
  import { BrowserRPCHandler } from "./sync/BrowserRPCHandler";

  const nodeTypes = {
    custom: CustomNode,
    subflow: SubflowNode,
    join: JoinNode,
  };

  // 2. Istanziamo il nostro "Controller/Model"
  // Grazie a Svelte 5, le sue proprietà interne $state saranno reattive qui dentro!
  const diagram = new Diagram();

  // Esponiamo il diagram via context per SubflowNode e altri componenti
  setContext<DiagramCore>("diagram", diagram);

  // --- SVELTE 5: Stato derivato per abilitare/disabilitare i pulsanti ---
  // Ora peschiamo direttamente dall'istanza diagram
  let selectedNodes = $derived(diagram.nodes.filter((n) => n.selected));
  let selectedEdges = $derived(diagram.edges.filter((e) => e.selected));

  // TODO: serve questo codice? let isNodeSelected = $derived(selectedNodes.length === 1);
  let hasSelection = $derived(
    selectedNodes.length > 0 || selectedEdges.length > 0,
  );

  let activeNode = $derived(
    selectedNodes.length === 1 ? selectedNodes[0] : null,
  );

  let isSidebarOpen = $state(false);
  let canvasRef: HTMLDivElement;

  // Auto-apertura quando si seleziona un nodo
  $effect(() => {
    if (activeNode) {
      isSidebarOpen = true;
    }
  });

  // Connessione WebSocket per gestire richieste RPC dal MCP server
  let syncClient: BrowserRPCHandler;

  $effect(() => {
    syncClient = new BrowserRPCHandler(diagram);
    syncClient.connect();
    return () => syncClient.disconnect();
  });

  function getSpawnPosition() {
    // Troviamo il centro della finestra e lo convertiamo in coordinate del canvas
    const center = screenToFlowPosition({
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
    });

    // Aggiungiamo un offset randomico tra -30px e +30px per non sovrapporli esattamente
    return {
      x: center.x + (Math.random() * 60 - 30),
      y: center.y + (Math.random() * 60 - 30),
    };
  }

  function handleAddSubGraph() {
    const pos = getSpawnPosition();
    diagram.addSubGraph(pos.x, pos.y);
  }

  function deleteSelectedElements() {
    if (selectedNodes.length > 0)
      diagram.deleteNodes(selectedNodes.map((n) => n.id));

    if (selectedEdges.length > 0)
      diagram.deleteEdges(selectedEdges.map((e) => e.id));
  }

  async function handleExportPng() {
    if (!canvasRef) return;

    const dataUrl = await toPng(canvasRef, {
      backgroundColor: "#ffffff",
      filter: (element) => {
        return (
          !element.classList?.contains("toolbar") &&
          !element.classList?.contains("svelte-flow__controls")
        );
      },
    });

    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = "diagram.png";
    link.click();
  }

  async function handleConversion() {
    const nnTree = new NNTree(diagram);
    const data = nnTree.toJson();

    // Controlla se showSaveFilePicker esiste (Chrome/Edge)
    if ("showSaveFilePicker" in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: "nnTree.json",
          types: [{ accept: { "application/json": [".json"] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(data);
        await writable.close();
        return;
      } catch (e) {
        console.warn("L'utente ha chiuso la finestra o c'è stato un errore.");
        return;
      }
    }

    // Fallback per Firefox e browser vecchi
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "nnTree.json";
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="editor-layout">
  <div class="canvas-container" bind:this={canvasRef}>
    <SvelteFlow
      bind:nodes={diagram.nodes}
      bind:edges={diagram.edges}
      {nodeTypes}
      defaultEdgeOptions={{
        markerEnd: { type: MarkerType.ArrowClosed },
      }}
      isValidConnection={(conn: Connection | Edge) =>
        checkValidConnection(diagram, conn)}
      onnodedragstop={(payload) => {
        let newNodes = onNodeDragStop(
          payload,
          diagram.nodes,
          getIntersectingNodes,
          getInternalNode,
        );
        if (newNodes !== undefined) diagram.nodes = newNodes;
      }}
      fitView
    >
      <Background />
      <Controls />
      <Panel position="top-left" class="toolbar">
        <button onclick={() => handleSaveModel(diagram)} class="toolbar-btn"
          >💾 Salva</button
        >
        <button
          onclick={() => {
            handleLoadModel(diagram);
            isSidebarOpen = false;
          }}
          class="toolbar-btn">📂 Carica</button
        >
        <button onclick={handleConversion} class="toolbar-btn"
          >📦 Converti in Python</button
        >
        <button onclick={handleExportPng} class="toolbar-btn"
          >🖼️ Esporta PNG</button
        >
        <button onclick={handleAddSubGraph} class="toolbar-btn"
          >📦 Aggiungi SubGraph</button
        >
        <button
          onclick={deleteSelectedElements}
          disabled={!hasSelection}
          class:danger={hasSelection}
        >
          ❌ Elimina
        </button>
      </Panel>
      <Panel position="top-right">
        <button
          class="toggle-sidebar-btn"
          onclick={() => (isSidebarOpen = !isSidebarOpen)}
        >
          {isSidebarOpen ? "Nascondi Proprietà" : "⚙️ Mostra Proprietà"}
        </button>
      </Panel>
    </SvelteFlow>
  </div>

  <Sidebar
    {diagram}
    selectedNode={activeNode}
    isOpen={isSidebarOpen}
    onClose={() => (isSidebarOpen = false)}
    {getSpawnPosition}
  />
</div>

<style>
  @import "./styles/flowcanvas.css";
</style>
