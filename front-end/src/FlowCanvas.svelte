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

  import {NNTree} from "./conversion/nnTree";

  // 1. Importiamo la classe Diagram
  import { Diagram } from "./Diagram.svelte"; // Modifica il path se necessario

  const nodeTypes = {
    custom: CustomNode,
    subflow: SubflowNode,
    join: JoinNode,
  };

  // 2. Istanziamo il nostro "Controller/Model"
  // Grazie a Svelte 5, le sue proprietà interne $state saranno reattive qui dentro!
  const diagram = new Diagram();

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

  // Auto-apertura quando si seleziona un nodo
  $effect(() => {
    if (activeNode) {
      isSidebarOpen = true;
    }
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

  function handleConversion(){
    const nnTree = new NNTree(diagram);
    console.log("NNTree:", nnTree.toJson());
  }
  
</script>

<div class="editor-layout">
  <div class="canvas-container">
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
