import { Stereotype } from "../stereotype";
import { Diagram } from "../Diagram.svelte";
import { type Node, type Edge } from "@xyflow/svelte";
/**
 * A tree node representing either a collapsed sequential block or a branching point.
 * - If sequential: contains multiple layers that are collapsed into one logical unit
 * - If fork/join: contains branches that split or merge the data flow
 */

export class NNTree {
  public root: NNTreeNode;
  public lossNode: ModuleData | null = null;

  //TODO: ADJUST
  constructor(diagram: Diagram) {
    const inputNodes: Node[] = diagram.nodes.filter(n => n.data.stereotype === "Input");
    if (inputNodes.length !== 1) {
      throw new Error("Expected exactly one input node, but found " + inputNodes.length);
    }
    this.root = this.processNode(inputNodes[0], diagram, new Set());
  }

  private createSequential(node: Node, diagram: Diagram, visited: Set<string>, childs: Node[]): NNTreeNode {
    // sequential
    let seq = [];
    seq.push({
      type: "module",
      moduleId: node.id,
      name: node.data.name,
      stereotype: node.data.stereotype,
      params: node.data.params
    } as ModuleData)
    do {
      let child = childs[0];
      visited.add(child.id);
      childs = diagram.getChilds(child.id);
      // TODO: potrebbe essere la loss
      if (childs.length === 0) {
        // Loss
        this.lossNode = {
          type: "module",
          moduleId: child.id,
          name: child.data.name,
          stereotype: child.data.stereotype,
          params: child.data.params
        } as ModuleData;
        break
      }
      seq.push({
        type: "module",
        moduleId: child.id,
        name: child.data.name,
        stereotype: child.data.stereotype,
        params: child.data.params
      } as ModuleData
      )

    } while (childs.length === 1);

    let next_tree_nodes: NNTreeNode[] = [];
    for (const child of childs) {
      next_tree_nodes.push(this.processNode(child, diagram, visited));
    }
    return NNTreeNode.fromParts(node.id, next_tree_nodes, {
      type: "sequential",
      layers: seq,
    });
  }

  private processNode(node: Node, diagram: Diagram, visited: Set<string>): NNTreeNode {
    if (visited.has(node.id)) {
      throw new Error("Node with id " + node.id + "is visited, there is a loop");
    }
    visited.add(node.id);

    let childs = diagram.getChilds(node.id);
    if (childs.length === 1) {
      return this.createSequential(node, diagram, visited, childs);
    } else if (childs.length > 1) {
      // fork
      let next_tree_nodes: NNTreeNode[] = [];
      for (const child of childs) {
        next_tree_nodes.push(this.processNode(child, diagram, visited));
      }
      return NNTreeNode.fromParts(node.id, next_tree_nodes, {
        type: "module",
        moduleId: node.id,
        name: node.data.name,
        stereotype: node.data.stereotype,
        params: node.data.params
      } as ModuleData);
    } else {
      // Loss
      this.lossNode = {
        type: "module",
        moduleId: node.id,
        name: node.data.name,
        stereotype: node.data.stereotype,
        params: node.data.params
      } as ModuleData;
    }
  }

  public toJson(): string {
    return JSON.stringify(this, null, 2);
  }
}

export class NNTreeNode {
  public id: string;
  public children: NNTreeNode[] = [];
  public data: SequentialData | ModuleData;
  public inputNodes: string[] = []; // For tracking which nodes feed into this node

  static fromParts(id: string, children: NNTreeNode[], data: SequentialData | ModuleData): NNTreeNode {
    return {
      id: id,
      children: children,
      data: data,
      inputNodes: [] as string[],
    } as NNTreeNode;
  }

  constructor(id: string, diagram: Diagram, sequentialData?: SequentialData) {
    this.id = id;

    // Trova i nodi figli direttamente connessi a questo nodo
    let nextNodes: string[] = diagram.edges
      .filter(e => e.source === id)
      .map(e => e.target);

    // Se ha piu nextNodes si tratta di un fork e quindi aggiungiamo direttamente a this.children
    // Se ha un solo nextNode bisogna appendere al sequential e verificare che non esista gia
    // TODO: Caso sequential senza figli
    if (nextNodes.length === 1) {

      // Prendiamo il nodo da diagram e verifichiamo che esista
      const currentNextNode: Node | undefined = diagram.nodes.find(n => n.id === nextNodes[0]);
      if (!currentNextNode) {
        throw new Error("Node " + nextNodes[0] + " not found in diagram.");
      }

      // Se ci troviamo gia in un blocco sequenziale appendiamo al sequential esistente
      if (sequentialData) {
        this.data = {
          type: "module",
          moduleId: nextNodes[0],
          name: currentNextNode.data.name,
          stereotype: currentNextNode.data.stereotype,
          params: currentNextNode.data.params
        } as ModuleData;
        sequentialData.layers.push(this.data);
      }
      // Altrimenti creo un nuovo blocco sequenziale con questo nodo come primo layer
      else {
        this.data = {
          type: "sequential",
          layers: [{
            type: "module",
            moduleId: nextNodes[0],
            name: currentNextNode.data.name,
            stereotype: currentNextNode.data.stereotype,
            params: currentNextNode.data.params
          }]
        } as SequentialData;
      }
    } else {
      // Se ha piu di un nextnode si tratta di un fork e quindi aggiungiamo direttamente i figli
      if (nextNodes.length > 1) {
        this.children = nextNodes.map(targetId => new NNTreeNode(targetId, diagram));
      }

      // In ogni caso, non essendo un sequential, salviamo il noto corrente come modulo
      const currentNode: Node | undefined = diagram.nodes.find(n => n.id === id);
      if (!currentNode) {
        throw new Error("Node " + id + " not found in diagram (type2).");
      }
      this.data = {
        type: "module",
        moduleId: id,
        name: currentNode.data.name,
        stereotype: currentNode.data.stereotype,
        params: currentNode.data.params
      } as ModuleData;
    }
  }

  addChild(child: NNTreeNode): void {
    this.children.push(child);
  }

  addInputNode(nodeId: string): void {
    if (!this.inputNodes.includes(nodeId)) {
      this.inputNodes.push(nodeId);
    }
  }

  removeChild(childId: string): boolean {
    const index = this.children.findIndex((c) => c.id === childId);
    if (index !== -1) {
      this.children.splice(index, 1);
      return true;
    }
    return false;
  }

  isSequential(): boolean {
    return (this.data as SequentialData).type === "sequential";
  }

  //   isFork(): boolean {
  //     return (this.data as ForkData).type === "fork";
  //   }

  //   isJoin(): boolean {
  //     return (this.data as JoinData).type === "join";
  //   }

  isModule(): boolean {
    return (this.data as ModuleData).type === "module";
  }

  //   isEmpty(): boolean {
  //     return (this.data as any).type === "empty";
  //   }
}

/**
 * Data structures for different node types
 */

// Sequential block containing multiple layers
export interface SequentialData {
  type: "sequential";
  layers: ModuleData[];
}

// // Fork node with multiple output branches
// export interface ForkData {
//   type: "fork";
//   branches: NNTreeNode[]; // Multiple paths from this point
// }

// // Join node with multiple input branches
// export interface JoinData {
//   type: "join";
//   joinType: string;
//   inputNodes: string[]; // IDs of nodes being joined
//   outputChannel?: number; // Track output channel count for validation
// }

// // Empty data for initial node state
// export interface EmptyData {
//   type: "empty";
// }

// Single module layer
export interface ModuleData {
  type: "module";
  moduleId: string;
  name: string;
  stereotype: string;
  params: any;
}
