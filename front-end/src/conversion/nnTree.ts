import { Diagram } from "../Diagram.svelte";
import { type Node } from "@xyflow/svelte";

export class NNTree {
  public nodes: Map<string, NNTreeNode>;
  public root: string;
  public lossNode: ModuleData | null = null;

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

  private getPythonClassName(diagram: Diagram, node: Node): string {
    const stereo = diagram.getStereotype(node.data.stereotype);
    return stereo?.pythonClassName || "";
  }

  private getTaskType(diagram: Diagram, node: Node): string {
    const stereo = diagram.getStereotype(node.data.stereotype);
    return stereo?.taskType || "";
  }

  private createSequential(node: Node, diagram: Diagram, visited: Set<string>, childs: Node[]): string {
    let seq = [];
    seq.push({
      type: "module",
      name: node.data.name,
      stereotype: node.data.stereotype,
      pythonClassName: this.getPythonClassName(diagram, node),
      params: node.data.params
    } as ModuleData)
    do {
      let child = childs[0];
      let parents = diagram.getParents(child.id);
      if (parents.length > 1) {
        break;
      }
      visited.add(child.id);
      childs = diagram.getChilds(child.id);
      if (childs.length === 0) {
        this.lossNode = {
          type: "module",
          moduleId: child.id,
          name: child.data.name,
          stereotype: child.data.stereotype,
          pythonClassName: this.getPythonClassName(diagram, child),
          taskType: this.getTaskType(diagram, child),
          params: child.data.params
        } as ModuleData;
        break
      }
      seq.push({
        type: "module",
        name: child.data.name,
        stereotype: child.data.stereotype,
        pythonClassName: this.getPythonClassName(diagram, child),
        params: child.data.params
      } as ModuleData)

    } while (childs.length === 1);

    let next_tree_nodes: string[] = [];
    for (const child of childs) {
      let nn_node = this.processNode(child, diagram, visited);
      if (nn_node !== undefined)
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
      return this.handleJoin(node, diagram, visited);
    }

    let childs = diagram.getChilds(node.id);
    if (childs.length === 1) {
      return this.createSequential(node, diagram, visited, childs);
    } else if (childs.length > 1) {
      let next_tree_nodes: string[] = [];
      for (const child of childs) {
        let nn_node = this.processNode(child, diagram, visited);
        if (nn_node !== undefined)
          next_tree_nodes.push(nn_node);
      }
      this.nodes.set(node.id, new NNTreeNode(node.id, next_tree_nodes, {
        type: "module",
        name: node.data.name,
        stereotype: node.data.stereotype,
        pythonClassName: this.getPythonClassName(diagram, node),
        params: node.data.params
      } as ModuleData));
      return node.id;
    } else {
      this.lossNode = {
        type: "module",
        name: node.data.name,
        stereotype: node.data.stereotype,
        pythonClassName: this.getPythonClassName(diagram, node),
        taskType: this.getTaskType(diagram, node),
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
  public children: string[] = [];
  public data: SequentialData | ModuleData | JoinData;

  constructor(id: string, children: string[], data: SequentialData | ModuleData | JoinData) {
    this.id = id;
    this.children = children;
    this.data = data;
  }


  addChild(child: string): void {
    this.children.push(child);
  }

  removeChild(childId: string): boolean {
    const index = this.children.findIndex((c) => c === childId);
    if (index !== -1) {
      this.children.splice(index, 1);
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

export interface SequentialData {
  type: "sequential";
  layers: ModuleData[];
}

export interface JoinData {
  type: "join";
  name: string;
  stereotype: string;
  params: any;
}

export interface ModuleData {
  type: "module";
  name: string;
  stereotype: string;
  pythonClassName?: string;
  taskType?: string;
  params: any;
}
