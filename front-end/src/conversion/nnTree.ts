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
    this.root = new NNTreeNode(inputNodes[0].id);
    this.processNode(this.root, null, new Set());
  }

  private processNode(node: Node, parent: NNTreeNode | null, visited: Set<string>): void {

  }
}

export class NNTreeNode {
  public id: string;
  public children: NNTreeNode[];
  public data: SequentialData | ForkData | JoinData | ModuleData | EmptyData;
  public inputNodes: string[]; // For tracking which nodes feed into this node

  constructor(id: string, parentId: string | null = null) {
    this.id = id;
    this.children = [];
    this.data = { type: "empty" };
    this.inputNodes = [];
  }

  processNode(node: Node, parent: NNTreeNode | null, visited: Set<string>): void {
    if (visited.has(node.id)) {
      throw new Error(`Cycle detected at node ${node.id}. This should not happen in a well-formed diagram.`);
    }
    visited.add(node.id);

    const stereotypeName = node.data.stereotype;
    const stereotype = Stereotype.getByName(stereotypeName);
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

  isFork(): boolean {
    return (this.data as ForkData).type === "fork";
  }

  isJoin(): boolean {
    return (this.data as JoinData).type === "join";
  }

  isModule(): boolean {
    return (this.data as ModuleData).type === "module";
  }

  isEmpty(): boolean {
    return (this.data as any).type === "empty";
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

// Fork node with multiple output branches
export interface ForkData {
  type: "fork";
  branches: NNTreeNode[]; // Multiple paths from this point
}

// Join node with multiple input branches
export interface JoinData {
  type: "join";
  joinType: Stereotype;
  inputNodes: string[]; // IDs of nodes being joined
  outputChannel?: number; // Track output channel count for validation
}

// Empty data for initial node state
export interface EmptyData {
  type: "empty";
}

// Single module layer
export interface ModuleData {
  type: "module";
  moduleId: string;
  name: string;
  stereotype: Stereotype;
  params: any;
}