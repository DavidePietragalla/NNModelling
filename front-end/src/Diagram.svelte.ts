import { type Node, type Edge } from "@xyflow/svelte";
import { Stereotype } from "./stereotype";

export class Diagram {
  public stereotypes: Stereotype[];

  // Usiamo $state per rendere reattivi gli array nativi di Svelte Flow
  public nodes: Node[] = $state.raw<Node[]>([]);
  public edges: Edge[] = $state.raw<Edge[]>([]);

  constructor() {
    // Boom! Una sola riga e hai caricato tutto.
    this.stereotypes = Stereotype.loadFromDirectory();

    // AUTO-SPAWN DELL'INPUT
    const inputStereotype = this.stereotypes.find(s => s.isInput);
    if (inputStereotype && this.nodes.length === 0) {
      // Lo posizioniamo in alto, più o meno al centro
      this.addModule(inputStereotype, window.innerWidth / 2 - 15, 50, {
        params: this.getDefaultParams(inputStereotype)
      });
    }
  }

  public getNodeById(id: string): Node | undefined {
    return this.nodes.find(n => n.id === id);
  }

  public getChilds(id: string): Node[] {
    let childsIds = this.edges.filter(e => e.source === id).map(e => e.target);
    let childs = [];
    childs.push(...this.nodes.filter(n => childsIds.find(c_id => c_id === n.id)));
    return childs;
  }

  public getParents(id: string): Node[] {
    let parentsIds = this.edges.filter(e => e.target === id).map(e => e.source);
    let parents = [];
    parents.push(...this.nodes.filter(n => parentsIds.find(c_id => c_id === n.id)));
    return parents;
  }

  public getStereotype(name: string): Stereotype | undefined {
    return this.stereotypes.find(s => s.name === name);
  }

  get layerStereotypes() { return this.stereotypes.filter(s => !s.isJoin); }
  get joinStereotypes() { return this.stereotypes.filter(s => s.isJoin); }

  // Aggiungere un modulo è banalissimo
  // Aggiungi un parametro opzionale 'customConfig'
  public addModule(
    stereotype: Stereotype,
    x: number,
    y: number,
    customConfig?: { name?: string, color?: string, width?: number, height?: number, params?: any }
  ) {
    // 1. Logica del nome: se l'utente ha scritto qualcosa usiamo quello,
    // altrimenti generiamo l'auto-nome (es. Tanh_0)
    let finalName = customConfig?.name;

    if (!finalName || finalName.trim() === "") {
      let counter = 0;
      while (this.nodes.some(n => n.data.name === `${stereotype.name}_${counter}`)) {
        counter++;
      }
      finalName = `${stereotype.name}_${counter}`;
    }

    // 2. Creiamo il nodo unendo i dati dello stereotipo con quelli del form
    const isInput = stereotype.isInput;
    const w = isInput ? 30 : (customConfig?.width || stereotype.view?.width || 140);
    const h = isInput ? 30 : (customConfig?.height || stereotype.view?.height || 60);

    const newNode = {
      id: crypto.randomUUID(),
      type: 'custom',
      position: { x, y },
      width: w,
      height: h,
      data: {
        stereotype: stereotype.name,
        name: finalName,
        color: customConfig?.color || stereotype.view?.color || '#ffffff',
        params: customConfig?.params ? JSON.parse(JSON.stringify(customConfig.params)) : {},
        // Passiamo i flag al frontend
        isInput: isInput,
        isLoss: stereotype.isLoss,
      }
    };
    // 3. Aggiungiamo il nodo allo stato
    this.nodes = [...this.nodes, newNode];
  }

  // Aggiungi questo metodo dentro la classe Diagram in Diagram.svelte.ts
  public addJoinNode(stereotype: Stereotype, x: number, y: number, config?: { name?: string, inputsCount?: number, params?: any }) {
    const id = `join_${crypto.randomUUID()}`;

    const newJoinNode: Node = {
      id,
      type: "join",
      position: { x, y },
      data: {
        stereotype: stereotype.name,
        name: stereotype.name,
        inputsCount: config?.inputsCount || 2,
        color: stereotype.view?.color || "#333",
        params: config?.params ? JSON.parse(JSON.stringify(config.params)) : {}
      }
    };

    this.nodes = [...this.nodes, newJoinNode];
  }

  // Da mettere dentro Diagram
  public addSubGraph(x: number, y: number) {
    const id = `subflow_${Date.now()}`;
    let newSubgraph: Node = {
      id,
      type: "subflow",
      position: { x, y },
      data: {
        label: `${id}`, 
        isCollapsed: false,
        onToggle: (id: string, collapse: boolean) => this.toggleSubflow(id, collapse),
        oldWidth: 400,
        oldHeight: 300,
        onResizeEnd: (nodeId: string, w: number, h: number) => {
          this.nodes = this.nodes.map(n => {
            if (n.id === nodeId && !n.data.isCollapsed) {
              return {
                ...n,
                data: {
                  ...n.data,
                  oldWidth: w,
                  oldHeight: h
                }
              } as Node;
            }
            return n;
          });
        }
      },
      width: 400,
      height: 300
    }
    this.nodes = [...this.nodes, newSubgraph];
  }

  public updateModule(id: string, config: { name?: string, label?: string, color?: string, width?: number, height?: number, params?: any, stereotype?: string }) {
    this.nodes = this.nodes.map(node => {
      if (node.id === id) {
        return {
          ...node,
          width: config.width ?? node.width,
          height: config.height ?? node.height,
          data: {
            ...node.data,
            name: config.name ?? node.data.name,
            label: config.label ?? node.data.label,
            color: config.color ?? node.data.color,
            stereotype: config.stereotype ?? node.data.stereotype,
            params: config.params ? JSON.parse(JSON.stringify(config.params)) : node.data.params,
            oldWidth: config.width ?? node.data.oldWidth,
            oldHeight: config.height ?? node.data.oldHeight,
          }
        };
      }
      return node;
    });
  }

  // Eliminare un singolo nodo si appoggia direttamente alla funzione multipla
  public deleteNode(id: string) {
    this.deleteNodes([id]);
  }

  public deleteNodes(ids: string[]) {
    // 1. Usiamo un Set per ricerche iper-veloci e una Mappa di TUTTI i nodi prima dell'eliminazione
    const nodesToDelete = new Set(ids);
    const allNodesMap = new Map(this.nodes.map(n => [n.id, n]));

    // 2. Filtriamo via i nodi eliminati e ricalcoliamo la parentela
    this.nodes = this.nodes
      .filter((n) => !nodesToDelete.has(n.id))
      .map((n) => {
        // Se il nodo non ha un padre, o il suo padre NON è tra quelli eliminati, è al sicuro.
        if (!n.parentId || !nodesToDelete.has(n.parentId)) {
          return n;
        }

        // Il padre diretto è stato eliminato! Iniziamo la ricerca dell'antenato superstite.
        let currentAncestorId: string | undefined = n.parentId;
        let accumulatedX = n.position.x;
        let accumulatedY = n.position.y;

        // Risaliamo l'albero genealogico finché il genitore in esame fa parte di quelli eliminati
        while (currentAncestorId && nodesToDelete.has(currentAncestorId)) {
          const deadAncestor = allNodesMap.get(currentAncestorId);
          if (!deadAncestor) break; // Fallback di sicurezza

          // Sommiamo l'offset del genitore eliminato per mantenere la posizione visiva intatta
          accumulatedX += deadAncestor.position.x;
          accumulatedY += deadAncestor.position.y;

          // Puntiamo al prossimo genitore (il "nonno")
          currentAncestorId = deadAncestor.parentId;
        }

        // Fine del ciclo: currentAncestorId è l'ID del primo antenato sopravvissuto,
        // oppure undefined se tutti gli antenati sono stati spazzati via.
        return {
          ...n,
          parentId: currentAncestorId,
          position: {
            x: accumulatedX,
            y: accumulatedY,
          },
        };
      });

    // 3. Facciamo pulizia degli edge collegati ai nodi eliminati
    this.edges = this.edges.filter(
      (e) => !nodesToDelete.has(e.source) && !nodesToDelete.has(e.target)
    );
  }

  public deleteEdges(edgesIds: string[]) {
    this.edges = this.edges.filter((e) => edgesIds.find(id => id == e.id) === undefined);
  }

  public deleteEdge(edgeId: string) {
    this.edges = this.edges.filter((e) => e.id !== edgeId);
  }

  toggleSubflow(parentId: string, willCollapse: boolean) {
    for (const child of this.nodes.filter(n => n.parentId === parentId)) {
      if (child.type === "subflow") {
        this.toggleSubflow(child.id, willCollapse);
      } 
    }
    this.nodes = this.nodes.map((node) => {
      if (node.parentId === parentId) {
        return { ...node, hidden: willCollapse };
      }
      
      if (node.id === parentId) {
        return {
          ...node,
          width: willCollapse ? 250 : node.data.oldWidth,
          height: willCollapse ? 50 : node.data.oldHeight,
          data: {
            ...node.data,
            isCollapsed: willCollapse
          }
        } as Node;
      } 
      
      return node;
    });

    const childNodeIds = this.nodes
    .filter((node) => node.parentId === parentId)
    .map((node) => node.id);
    this.edges = this.edges.map((edge) => {
      const isConnectedToChild = childNodeIds.includes(edge.source) || childNodeIds.includes(edge.target);

      if (isConnectedToChild) {
        return {
          ...edge,
          hidden: willCollapse
        };
      }

      return edge;
    });
  }

  private getDefaultParams(stereotype: Stereotype): Record<string, any> {
    if (!stereotype.parameters) {
      return {};
    }

    return Object.fromEntries(
      Object.entries(stereotype.parameters).map(([key, paramDef]) => [
        key,
        // INVECE DI RITORNARE SOLO IL VALORE...
        // paramDef.default 

        // ... RITORNIAMO L'OGGETTO COMPLETO:
        { value: paramDef.default, position: paramDef.position }
      ])
    );
  }

  public exportToJson(): string {
    // SvelteFlow mantiene le strutture dati pulite.
    // Possiamo serializzare direttamente i nodi e gli edges.
    const exportData = {
      nodes: this.nodes,
      edges: this.edges,
    };

    // Il '2' serve per formattare il JSON con indentazione (più leggibile)
    return JSON.stringify(exportData, null, 2);
  }

  public importFromJson(jsonString: string) {
    try {
      const parsedData = JSON.parse(jsonString);

      // Validazione base per assicurarci che il file sia corretto
      if (Array.isArray(parsedData.nodes) && Array.isArray(parsedData.edges)) {
        // Sovrascriviamo lo stato reattivo. 
        // SvelteFlow si aggiornerà automaticamente!
        this.nodes = parsedData.nodes;
        this.edges = parsedData.edges;
      } else {
        throw new Error("Il file JSON non contiene un formato valido (nodi o edges mancanti).");
      }
    } catch (error) {
      console.error("Errore durante l'importazione del modello:", error);
      alert("Errore: Impossibile caricare il file. Verifica che sia un JSON valido.");
    }
  }
}
