import { type Node, type Edge } from "@xyflow/svelte";
import { Stereotype, type StereotypeJson } from "../stereotype";

export class TestDiagram {
  nodes: Node[] = [];
  edges: Edge[] = [];
  stereotypes: Stereotype[] = [];

  constructor(nodes: Node[], edges: Edge[], stereotypes: Stereotype[]) {
    this.nodes = nodes;
    this.edges = edges;
    this.stereotypes = stereotypes;
  }

  getChilds(id: string): Node[] {
    const childIds = this.edges.filter((e) => e.source === id).map((e) => e.target);
    return this.nodes.filter((n) => childIds.includes(n.id));
  }

  getParents(id: string): Node[] {
    const parentIds = this.edges.filter((e) => e.target === id).map((e) => e.source);
    return this.nodes.filter((n) => parentIds.includes(n.id));
  }

  getStereotype(name: string): Stereotype | undefined {
    return this.stereotypes.find((s) => s.name === name);
  }

  // --- Stub methods for type compatibility with Diagram ---
  getNodeById(id: string): Node | undefined {
    return this.nodes.find((n) => n.id === id);
  }

  get layerStereotypes() {
    return this.stereotypes.filter((s) => !s.isJoin);
  }
  get joinStereotypes() {
    return this.stereotypes.filter((s) => s.isJoin);
  }

  addModule() {}
  addJoinNode() {}
  addSubGraph() {}
  updateModule() {}
  deleteNode() {}
  deleteNodes(_ids: string[]) {}
  deleteEdges(_ids: string[]) {}
  deleteEdge(_id: string) {}
  toggleSubflow() {}
  exportToJson(): string {
    return JSON.stringify({ nodes: this.nodes, edges: this.edges });
  }
  importFromJson() {}
}

export function buildStereotype(
  name: string,
  overrides?: Partial<StereotypeJson>,
): Stereotype {
  const defaults: StereotypeJson = {
    category: name,
    pythonClassName: name === "Input" ? "None" : `nn.${name}`,
    view: { color: "#ccc", width: 100, height: 50 },
    params: {},
  };

  const isJoin = overrides?.category === "Join" || false;
  const filePath = isJoin
    ? `test://Stereotypes/Joins/${name}.json`
    : `test://Stereotypes/Modules/${name}.json`;

  return new Stereotype(filePath, { ...defaults, ...overrides });
}

export function param(value: string, position?: "top" | "bottom") {
  return position ? { value, position } : { value };
}
