import { Diagram } from "../Diagram.svelte";
import { type Node } from "@xyflow/svelte";
/**
 * A tree node representing either a collapsed sequential block or a branching point.
 * - If sequential: contains multiple layers that are collapsed into one logical unit
 * - If fork/join: contains branches that split or merge the data flow
 */

export class NNTree {
  public nodes: Map<string, NNTreeNode>;
  public root: string;
  public lossNode: ModuleData | null = null;

  //TODO: ADJUST
  constructor(diagram: Diagram) {
    this.nodes = new Map();
    const inputNodes: Node[] = diagram.nodes.filter(n => n.data.stereotype === "Input");
    if (inputNodes.length !== 1) {
      throw new Error("Expected exactly one input node, but found " + inputNodes.length);
    }
    let new_root = this.processNode(inputNodes[0], diagram, new Set());
    if (new_root === undefined) throw new Error("root is undefined");
    this.root = new_root;
  }

  private createSequential(node: Node, diagram: Diagram, visited: Set<string>, childs: Node[]): string {
    // sequential
    let seq = [];
    seq.push({
      type: "module",
      name: node.data.name,
      stereotype: node.data.stereotype,
      params: node.data.params
    } as ModuleData)
    do {
      let child = childs[0];
      let parents = diagram.getParents(child.id);
      if (parents.length > 1) {
        // Join
        break;
      }
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
        name: child.data.name,
        stereotype: child.data.stereotype,
        params: child.data.params
      } as ModuleData
      )

    } while (childs.length === 1);

    let next_tree_nodes: string[] = [];
    for (const child of childs) {
      let nn_node = this.processNode(child, diagram, visited);
      if (nn_node !== undefined) // TODO: cambia
        next_tree_nodes.push(nn_node);
    }
    this.nodes.set(node.id, new NNTreeNode(node.id, next_tree_nodes, {
      type: "sequential",
      layers: seq,
    }));
    return node.id;

  }

  private handleJoin(node: Node, diagram: Diagram, visited: Set<string>): string {
    let childs = diagram.getChilds(node.id);
    let next_tree_nodes: string[] = [];
    for (const child of childs) {
      let nn_node = this.processNode(child, diagram, visited);
      if (nn_node !== undefined)
        next_tree_nodes.push(nn_node);
    }
    this.nodes.set(node.id, new NNTreeNode(node.id, next_tree_nodes, {
      type: "join",
      name: node.data.name,
      stereotype: node.data.stereotype,
      params: node.data.params
    } as JoinData));
    return node.id;

  }

  private processNode(node: Node, diagram: Diagram, visited: Set<string>): string | undefined {
    if (visited.has(node.id)) {
      console.warn("Node with id " + node.id + "is visited, there is a loop");
      return node.id;
    }
    visited.add(node.id);

    let parents = diagram.getParents(node.id);
    if (parents.length > 1) {
      // Join
      return this.handleJoin(node, diagram, visited);
    }

    let childs = diagram.getChilds(node.id);
    if (childs.length === 1) {
      return this.createSequential(node, diagram, visited, childs);
    } else if (childs.length > 1) {
      // fork
      let next_tree_nodes: string[] = [];
      for (const child of childs) {
        let nn_node = this.processNode(child, diagram, visited);
        if (nn_node !== undefined) // TODO: cambia
          next_tree_nodes.push(nn_node);
      }
      this.nodes.set(node.id, new NNTreeNode(node.id, next_tree_nodes, {
        type: "module",
        name: node.data.name,
        stereotype: node.data.stereotype,
        params: node.data.params
      } as ModuleData));
      return node.id;
    } else {
      // Loss
      this.lossNode = {
        type: "module",
        name: node.data.name,
        stereotype: node.data.stereotype,
        params: node.data.params,
      } as ModuleData;
      return;
    }
  }

  public toJson(): string {
    const serializableObject = {
      root: this.root,
      lossNode: this.lossNode,
      nodes: Object.fromEntries(this.nodes)
    };

    return JSON.stringify(serializableObject, null, 2);
  }
}

export class NNTreeNode {
  public id: string;
  public childrens: string[] = [];
  public data: SequentialData | ModuleData | JoinData;

  constructor(id: string, childrens: string[], data: SequentialData | ModuleData | JoinData) {
    this.id = id;
    this.childrens = childrens;
    this.data = data;
  }


  addChild(child: string): void {
    this.childrens.push(child);
  }

  removeChild(childId: string): boolean {
    const index = this.childrens.findIndex((c) => c === childId);
    if (index !== -1) {
      this.childrens.splice(index, 1);
      return true;
    }
    return false;
  }

  isSequential(): boolean {
    return (this.data as SequentialData).type === "sequential";
  }

  isJoin(): boolean {
    return (this.data as JoinData).type === "join";
  }

  isModule(): boolean {
    return (this.data as ModuleData).type === "module";
  }

}

/**
 * Data structures for different node types
 */

// Sequential block containing multiple layers
export interface SequentialData {
  type: "sequential";
  layers: ModuleData[];
}

// // Join node with multiple input branches
export interface JoinData {
  type: "join";
  name: string;
  stereotype: string;
  params: any;
}

// Single module layer
export interface ModuleData {
  type: "module";
  name: string;
  stereotype: string;
  params: any;
}
