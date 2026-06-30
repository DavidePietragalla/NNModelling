import type { DiagramCore } from "../core/DiagramCore";
import { type Node } from "@xyflow/svelte";

export class NNTree {
  public nodes: Map<string, NNTreeNode>;
  public root: string;
  public lossNode: ModuleData | null = null;

  constructor(diagram: DiagramCore) {
    this.nodes = new Map();
    const inputNodes: Node[] = diagram.nodes.filter(n => n.data.stereotype === "Input");
    if (inputNodes.length !== 1) {
      throw new Error("Expected exactly one input node, but found " + inputNodes.length);
    }
    let new_root = this.processNode(inputNodes[0], diagram, new Set());
    if (new_root === undefined) throw new Error("root is undefined");
    this.root = new_root;
  }

  private getPythonClassName(diagram: DiagramCore, node: Node): string {
    const stereo = diagram.getStereotype(node.data.stereotype as string);
    return stereo?.pythonClassName || "";
  }

  private getTaskType(diagram: DiagramCore, node: Node): string {
    const stereo = diagram.getStereotype(node.data.stereotype as string);
    return stereo?.taskType || "";
  }

  private isSubflowNode(node: Node): boolean {
    return node.type === "subflow";
  }

  private nodeToModule(node: Node, diagram: DiagramCore): ModuleData {
    return {
      type: "module",
      name: node.data.name,
      stereotype: node.data.stereotype,
      pythonClassName: this.getPythonClassName(diagram, node),
      params: node.data.params,
    } as ModuleData;
  }

  private compileSubflowGraph(diagram: DiagramCore, subflowId: string): SubflowGraph {
    const internalNodes = diagram.nodes.filter((n: any) => n.parentId === subflowId);
    if (internalNodes.length === 0) {
      console.warn("Subflow " + subflowId + " has no internal nodes");
      return { entryNode: "", nodes: {} };
    }
    const internalIds = new Set(internalNodes.map((n: any) => n.id));
    const internalEdges = diagram.edges.filter((e: any) =>
      internalIds.has(e.source) && internalIds.has(e.target),
    );

    // Kahn's topological sort
    const inDegree = new Map<string, number>();
    const adj = new Map<string, string[]>();
    for (const n of internalNodes) {
      inDegree.set(n.id, 0);
      adj.set(n.id, []);
    }
    for (const e of internalEdges) {
      adj.get(e.source)?.push(e.target);
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
    }

    const queue = internalNodes.filter((n: any) => inDegree.get(n.id) === 0).map((n: any) => n.id);
    const sorted: string[] = [];
    while (queue.length > 0) {
      const id = queue.shift()!;
      sorted.push(id);
      for (const target of adj.get(id) || []) {
        const d = (inDegree.get(target) || 1) - 1;
        inDegree.set(target, d);
        if (d === 0) queue.push(target);
      }
    }

    if (sorted.length !== internalNodes.length) {
      throw new Error("Subflow " + subflowId + " contains a cycle");
    }

    // Build input ordering for joins from edge targetHandles
    const targetInputs: Record<string, string[]> = {};
    for (const e of internalEdges) {
      const t = e.target;
      const h = parseInt((e.targetHandle || "in-0").replace("in-", ""));
      if (!targetInputs[t]) targetInputs[t] = [];
      targetInputs[t][h] = e.source;
    }

    const nodesMap: Record<string, InternalNodeData> = {};

    for (const id of sorted) {
      const n = internalNodes.find((m: any) => m.id === id)!;
      const children = adj.get(id) || [];

      if (this.isSubflowNode(n)) {
        const nested = this.compileSubflowGraph(diagram, n.id);
        nodesMap[id] = {
          type: "subflow",
          name: n.data.name,
          stereotype: n.data.stereotype,
          pythonClassName: this.getPythonClassName(diagram, n),
          params: n.data.params,
          children,
          entryNode: nested.entryNode,
          nodes: nested.nodes,
        };
      } else {
        const isJoinNode = n.type === "join"
          || diagram.getStereotype(n.data.stereotype as string)?.category === "Join";
        nodesMap[id] = {
          type: isJoinNode ? "join" : "module",
          name: n.data.name,
          stereotype: n.data.stereotype,
          pythonClassName: this.getPythonClassName(diagram, n),
          taskType: this.getTaskType(diagram, n),
          params: n.data.params,
          children,
          ...(isJoinNode ? { inputs: targetInputs[id] || [] } : {}),
        };
      }
    }

    return { entryNode: sorted[0], nodes: nodesMap };
  }

  private processSubflow(node: Node, diagram: DiagramCore, visited: Set<string>): string {
    const graph = this.compileSubflowGraph(diagram, node.id);

    const outerChilds = diagram.getChilds(node.id);
    const nextNodes: string[] = [];
    for (const child of outerChilds) {
      const nnNode = this.processNode(child, diagram, visited);
      if (nnNode !== undefined) nextNodes.push(nnNode);
    }

    this.nodes.set(
      node.id,
      new NNTreeNode(node.id, nextNodes, {
        type: "subflow",
        name: node.data.name,
        stereotype: node.data.stereotype,
        pythonClassName: this.getPythonClassName(diagram, node),
        params: node.data.params,
        entryNode: graph.entryNode,
        nodes: graph.nodes,
      } as SubflowData),
    );
    return node.id;
  }

  private createSequential(node: Node, diagram: DiagramCore, visited: Set<string>, childs: Node[]): string {
    let seq = [];
    seq.push(this.nodeToModule(node, diagram))
    do {
      let child = childs[0];
      let parents = diagram.getParents(child.id);
      if (parents.length > 1) {
        break;
      }
      if (this.isSubflowNode(child)) {
        break;
      }
      if (child.data.stereotype === "Fork") {
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
      seq.push(this.nodeToModule(child, diagram))

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

  private handleJoin(node: Node, diagram: DiagramCore, visited: Set<string>): string {
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
      pythonClassName: this.getPythonClassName(diagram, node),
      params: node.data.params
    } as JoinData));
    return node.id;

  }

  private processNode(node: Node, diagram: DiagramCore, visited: Set<string>): string | undefined {
    if (visited.has(node.id)) {
      console.warn("Node with id " + node.id + "is visited, there is a loop");
      return node.id;
    }
    visited.add(node.id);

    if (this.isSubflowNode(node)) {
      return this.processSubflow(node, diagram, visited);
    }

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
      this.nodes.set(node.id, new NNTreeNode(node.id, next_tree_nodes, this.nodeToModule(node, diagram)));
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
  public data: SequentialData | ModuleData | JoinData | SubflowData;

  constructor(id: string, children: string[], data: SequentialData | ModuleData | JoinData | SubflowData) {
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

  isSubflow(): boolean {
    return (this.data as SubflowData).type === "subflow";
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
  pythonClassName?: string;
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

export interface SubflowData {
  type: "subflow";
  name: string;
  stereotype: string;
  pythonClassName?: string;
  params: any;
  entryNode: string;
  nodes: Record<string, InternalNodeData>;
}

export interface SubflowGraph {
  entryNode: string;
  nodes: Record<string, InternalNodeData>;
}

export interface InternalNodeData {
  type: "module" | "join" | "subflow";
  name: string;
  stereotype: string;
  pythonClassName?: string;
  taskType?: string;
  params: any;
  children: string[];
  inputs?: string[];
  entryNode?: string;
  nodes?: Record<string, InternalNodeData>;
}
