/**
 * Full Pipeline Integration Test — Step 13
 *
 * Builds a complete MNIST classifier diagram programmatically using
 * DiagramCore's direct API (not MCP tool handlers), compiles it to
 * NNTree JSON, and validates the output structure.
 *
 * IMPORTANT: DiagramCore.addModule() returns void, not a Node.
 * To access the created node, use diagram.nodes[diagram.nodes.length - 1]
 * after each addModule call (nodes are always appended).
 *
 * This test validates the DiagramCore → NNTree pipeline end-to-end
 * without needing the MCP server or WebSocket running.
 */

import { describe, it, expect } from "vitest";
import path from "path";
import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { NNTree } from "@nnmodelling/front-end/conversion/nnTree";
import type { Node } from "@nnmodelling/front-end/core/types";

describe("Full pipeline integration", () => {
  /**
   * Helper to create a fresh DiagramCore with loaded stereotypes and
   * initialized nodes/edges arrays.
   */
  function createDiagram(): DiagramCore {
    const diagram = new DiagramCore();
    diagram.nodes = [];
    diagram.edges = [];

    // Resolve Stereotypes directory relative to this test file
    // __dirname = mcp-server/__tests__/
    // Stereotypes = <root>/Stereotypes/
    const stereotypesDir = path.resolve(__dirname, "../../Stereotypes");
    const stereotypes = StereotypeCore.loadFromDirectoryNode(stereotypesDir);
    diagram.initStereotypes(stereotypes);

    return diagram;
  }

  /**
   * Helper to get the last created node after calling addModule/addJoinNode.
   * Since addModule returns void, we capture the node count before creation
   * and retrieve the last appended node afterward.
   */
  function lastNode(diagram: DiagramCore): Node {
    return diagram.nodes[diagram.nodes.length - 1];
  }

  it("builds MNIST classifier: Input → Linear(784→128) → ReLU → Linear(128→10) → CrossEntropyLoss", () => {
    const diagram = createDiagram();

    // ── Get stereotypes ───────────────────────────────────────────────
    const inputStereo = diagram.getStereotype("Input");
    const linearStereo = diagram.getStereotype("Linear");
    const reluStereo = diagram.getStereotype("ReLU");
    const ceStereo = diagram.getStereotype("CrossEntropyLoss");

    expect(inputStereo).toBeTruthy();
    expect(linearStereo).toBeTruthy();
    expect(reluStereo).toBeTruthy();
    expect(ceStereo).toBeTruthy();

    // ── Build the graph programmatically ─────────────────────────────
    diagram.addModule(inputStereo!, 200, 50);
    const input = lastNode(diagram);

    diagram.addModule(linearStereo!, 200, 150, {
      params: { in_features: "784", out_features: "128" },
    });
    const lin1 = lastNode(diagram);

    diagram.addModule(reluStereo!, 200, 250);
    const relu = lastNode(diagram);

    diagram.addModule(linearStereo!, 200, 350, {
      params: { in_features: "128", out_features: "10" },
    });
    const lin2 = lastNode(diagram);

    diagram.addModule(ceStereo!, 200, 450);
    const loss = lastNode(diagram);

    // ── Connect edges using addEdge ───────────────────────────────────
    diagram.addEdge(input.id, lin1.id);
    diagram.addEdge(lin1.id, relu.id);
    diagram.addEdge(relu.id, lin2.id);
    diagram.addEdge(lin2.id, loss.id);

    // ── Verify diagram state ──────────────────────────────────────────
    expect(diagram.nodes).toHaveLength(5);
    expect(diagram.edges).toHaveLength(4);

    // Verify node data
    const lin1Node = diagram.getNodeById(lin1.id)!;
    expect((lin1Node.data as Record<string, unknown>).params).toEqual({
      in_features: "784",
      out_features: "128",
    });

    // ── Compile via NNTree ────────────────────────────────────────────
    const nntree = new NNTree(diagram);
    const json = nntree.toJson();
    const parsed = JSON.parse(json);

    // ── Validate NNTree output ────────────────────────────────────────
    expect(parsed.root).toBeTruthy();
    expect(typeof parsed.root).toBe("string");

    // Loss node
    expect(parsed.lossNode).toBeTruthy();
    expect(parsed.lossNode.stereotype).toBe("CrossEntropyLoss");
    expect(parsed.lossNode.taskType).toBe("classification");

    // Nodes map
    expect(parsed.nodes).toBeTruthy();
    const nodeKeys = Object.keys(parsed.nodes);
    expect(nodeKeys.length).toBeGreaterThanOrEqual(1);

    // The root node should be a "sequential" type containing Linear
    const rootNode = parsed.nodes[parsed.root];
    expect(rootNode).toBeTruthy();
    expect(rootNode.data.type).toBe("sequential");
    expect(rootNode.data.layers).toBeDefined();
    expect(Array.isArray(rootNode.data.layers)).toBe(true);

    // The sequential should contain Linear and ReLU layers
    const layerNames = rootNode.data.layers.map(
      (l: Record<string, unknown>) => l.stereotype
    );
    expect(layerNames).toContain("Linear");
    expect(layerNames).toContain("ReLU");
  });

  it("handles edge case: single Input node with no connections", () => {
    const diagram = createDiagram();
    const inputStereo = diagram.getStereotype("Input")!;
    diagram.addModule(inputStereo, 200, 50);

    // A diagram with only an Input node should fail to compile
    // (NNTree expects at least Input → something → Loss)
    expect(() => new NNTree(diagram)).toThrow();
  });

  it("handles edge case: repeated stereotype auto-naming", () => {
    const diagram = createDiagram();
    const linearStereo = diagram.getStereotype("Linear")!;

    diagram.addModule(linearStereo, 100, 100);
    const a = lastNode(diagram);
    diagram.addModule(linearStereo, 100, 200);
    const b = lastNode(diagram);

    // Names should auto-increment
    expect((a.data as Record<string, unknown>).name).toBe("Linear_0");
    expect((b.data as Record<string, unknown>).name).toBe("Linear_1");
  });

  it("emits events during diagram construction", () => {
    const diagram = createDiagram();
    const events: string[] = [];

    diagram.events.onAny((event) => {
      events.push(event.type);
    });

    const inputStereo = diagram.getStereotype("Input")!;
    const linearStereo = diagram.getStereotype("Linear")!;
    const ceStereo = diagram.getStereotype("CrossEntropyLoss")!;

    diagram.addModule(inputStereo, 200, 50);
    const input = lastNode(diagram);
    diagram.addModule(linearStereo, 200, 150);
    const lin = lastNode(diagram);
    diagram.addModule(ceStereo, 200, 250);
    const loss = lastNode(diagram);

    diagram.addEdge(input.id, lin.id);
    diagram.addEdge(lin.id, loss.id);

    // We should have captured: node_created (×3), graph_changed (×3),
    // edge_created (×2), graph_changed (×2)
    expect(events.filter((t) => t === "node_created").length).toBe(3);
    expect(events.filter((t) => t === "edge_created").length).toBe(2);
    expect(events.filter((t) => t === "graph_changed").length).toBe(5); // 3 nodes + 2 edges
  });

  it("verifies NNTree structure for a join-based diagram (skip connection)", () => {
    const diagram = createDiagram();
    const inputStereo = diagram.getStereotype("Input")!;
    const linearStereo = diagram.getStereotype("Linear")!;
    const reluStereo = diagram.getStereotype("ReLU")!;
    const additionStereo = diagram.getStereotype("Addition")!;
    const ceStereo = diagram.getStereotype("CrossEntropyLoss")!;

    // Build: Input → Linear1 → ReLU ─┐
    //                       Linear2 ──→ Addition → CrossEntropyLoss
    diagram.addModule(inputStereo!, 200, 50);
    const input = lastNode(diagram);

    diagram.addModule(linearStereo!, 150, 150, {
      params: { in_features: "784", out_features: "128" },
    });
    const lin1 = lastNode(diagram);

    diagram.addModule(reluStereo!, 150, 250);
    const relu = lastNode(diagram);

    diagram.addModule(linearStereo!, 350, 150, {
      params: { in_features: "784", out_features: "128" },
    });
    const lin2 = lastNode(diagram);

    diagram.addJoinNode(additionStereo!, 250, 350, {
      inputsCount: 2,
    });
    const add = lastNode(diagram);

    diagram.addModule(ceStereo!, 250, 450);
    const loss = lastNode(diagram);

    // Connect: Input forks to Lin1 and Lin2
    diagram.addEdge(input.id, lin1.id);
    diagram.addEdge(input.id, lin2.id);

    // Lin1 → ReLU
    diagram.addEdge(lin1.id, relu.id);

    // ReLU + Lin2 → Addition
    diagram.addEdge(relu.id, add.id, "out", "in-0");
    diagram.addEdge(lin2.id, add.id, "out", "in-1");

    // Addition → Loss
    diagram.addEdge(add.id, loss.id);

    expect(diagram.nodes).toHaveLength(6);
    expect(diagram.edges).toHaveLength(6);

    // Compile via NNTree
    const nntree = new NNTree(diagram);
    const json = nntree.toJson();
    const parsed = JSON.parse(json);

    // Validate NNTree
    expect(parsed.root).toBeTruthy();
    expect(parsed.lossNode.stereotype).toBe("CrossEntropyLoss");

    // Should have multiple tree nodes (Input fork → two branches)
    const nodeKeys = Object.keys(parsed.nodes);
    expect(nodeKeys.length).toBeGreaterThanOrEqual(3);

    // Find the Addition join node in the tree
    const joinNodes = nodeKeys.filter((k) => {
      const node = parsed.nodes[k];
      return node.data && node.data.type === "join";
    });

    expect(joinNodes.length).toBeGreaterThanOrEqual(1);
    expect(parsed.nodes[joinNodes[0]].data.stereotype).toBe("Addition");
  });
});
