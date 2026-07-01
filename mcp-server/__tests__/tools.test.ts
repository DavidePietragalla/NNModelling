/**
 * MCP Tool Integration Tests
 *
 * Tests core tools (create_node, set_parameter, select_nodes, delete_nodes)
 * using a real DiagramCore instance (in-memory, no MCP transport).
 *
 * Every test creates a fresh ServerContext with stereotype loading,
 * ensuring isolation between test cases.
 */

import { describe, it, expect, beforeEach } from "vitest";
import path from "path";
import { DiagramCore } from "@nnmodelling/front-end/core/DiagramCore";
import { StereotypeCore } from "@nnmodelling/front-end/core/StereotypeCore";
import { TransactionManager } from "../src/transaction";
import { HistoryManager } from "../src/history";
import { NNTree } from "@nnmodelling/front-end/conversion/nnTree";
import type { ServerContext } from "../src/server";

// Import tool modules
import * as graphTools from "../src/tools/graph";
import * as paramTools from "../src/tools/parameters";
import * as selectionTools from "../src/tools/selection";
import { StereotypeNotFoundError, NodeNotFoundError } from "../src/errors";

// ── Test Helper ───────────────────────────────────────────────────────────

function createTestContext(): ServerContext {
  const diagram = new DiagramCore();
  // Initialize plain arrays for headless Node.js operation (DiagramCore uses declare)
  diagram.nodes = [];
  diagram.edges = [];

  // Load stereotypes from the project's Stereotypes/ directory
  // Test file is at: mcp-server/__tests__/tools.test.ts
  // Stereotypes are at: <root>/Stereotypes/
  const stereotypesDir = path.resolve(__dirname, "../../Stereotypes");
  const stereotypes = StereotypeCore.loadFromDirectoryNode(stereotypesDir);
  diagram.initStereotypes(stereotypes);

  return {
    diagram,
    transactions: new TransactionManager(diagram),
    history: new HistoryManager(),
    pipeline: null as unknown as ServerContext["pipeline"],
    eventBuffer: [],
    lastEventCursor: 0,
  };
}

// ── Tests ─────────────────────────────────────────────────────────────────

describe("create_node", () => {
  let ctx: ServerContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  it("creates a Linear node with correct position and name", async () => {
    const result = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
      config: { params: { in_features: "784", out_features: "128" } },
    });

    expect(result.nodeId).toBeTruthy();
    expect(result.type).toBe("custom");
    expect(result.stereotype).toBe("Linear");

    const node = ctx.diagram.getNodeById(result.nodeId);
    expect(node).toBeTruthy();
    expect(node!.position).toEqual({ x: 100, y: 50 });
    expect(node!.data.name).toBe("Linear_0");
    expect(node!.data.stereotype).toBe("Linear");

    // Verify params were set
    const params = (node!.data as Record<string, unknown>).params as Record<string, unknown>;
    expect(params.in_features).toBe("784");
    expect(params.out_features).toBe("128");
  });

  it("throws StereotypeNotFoundError for unknown stereotype", async () => {
    await expect(
      graphTools.create_node.handler(ctx, {
        stereotype: "NonExistentLayer",
        position: { x: 0, y: 0 },
      })
    ).rejects.toThrow(StereotypeNotFoundError);
  });

  it("creates a join node (Addition) with type='join'", async () => {
    const result = await graphTools.create_node.handler(ctx, {
      stereotype: "Addition",
      position: { x: 300, y: 200 },
      config: { inputsCount: 3 },
    });

    expect(result.nodeId).toBeTruthy();
    expect(result.type).toBe("join");
    expect(result.stereotype).toBe("Addition");

    const node = ctx.diagram.getNodeById(result.nodeId);
    expect(node).toBeTruthy();
    expect(node!.type).toBe("join");
    expect(node!.position).toEqual({ x: 300, y: 200 });

    // Verify inputsCount on the node data
    const nodeData = node!.data as Record<string, unknown>;
    expect(nodeData.inputsCount).toBe(3);
  });

  it("tracks history after creation", async () => {
    const statusBefore = ctx.history.getStatus();
    expect(statusBefore.undoCount).toBe(0);

    await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 150, y: 150 },
    });

    const statusAfter = ctx.history.getStatus();
    expect(statusAfter.undoCount).toBe(1);
  });
});

describe("set_parameter", () => {
  let ctx: ServerContext;
  let linearNodeId: string;

  beforeEach(async () => {
    ctx = createTestContext();

    // Create a Linear node to manipulate
    const result = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
      config: { params: { in_features: "784", out_features: "128" } },
    });
    linearNodeId = result.nodeId;
  });

  it("updates a node parameter value", async () => {
    const result = await paramTools.set_parameter.handler(ctx, {
      nodeId: linearNodeId,
      key: "out_features",
      value: "256",
    });

    expect(result.nodeId).toBe(linearNodeId);
    expect(result.key).toBe("out_features");
    expect(result.previousValue).toBe("128");
    expect(result.currentValue).toBe("256");

    // Verify the node data was updated
    const node = ctx.diagram.getNodeById(linearNodeId)!;
    const params = (node.data as Record<string, unknown>).params as Record<string, unknown>;
    expect(params.out_features).toBe("256");
  });

  it("sets a new parameter key that didn't exist before", async () => {
    const result = await paramTools.set_parameter.handler(ctx, {
      nodeId: linearNodeId,
      key: "bias",
      value: "true",
    });

    expect(result.previousValue).toBeNull();
    expect(result.currentValue).toBe("true");

    const node = ctx.diagram.getNodeById(linearNodeId)!;
    const params = (node.data as Record<string, unknown>).params as Record<string, unknown>;
    expect(params.bias).toBe("true");
  });

  it("throws NodeNotFoundError for a bad nodeId", async () => {
    await expect(
      paramTools.set_parameter.handler(ctx, {
        nodeId: "non-existent-node-id",
        key: "in_features",
        value: "512",
      })
    ).rejects.toThrow(NodeNotFoundError);
  });
});

describe("select_nodes / clear_selection", () => {
  let ctx: ServerContext;
  let nodeIds: string[];

  beforeEach(async () => {
    ctx = createTestContext();

    // Create several nodes
    const lin = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
    });
    const relu = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 100, y: 150 },
    });
    const lin2 = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 250 },
    });
    nodeIds = [lin.nodeId, relu.nodeId, lin2.nodeId];
  });

  it("selects specific nodes", async () => {
    const result = await selectionTools.select_nodes.handler(ctx, {
      nodeIds: [nodeIds[0], nodeIds[2]],
      mode: "replace",
    });

    expect(result.selectedNodeIds).toEqual([nodeIds[0], nodeIds[2]]);

    // Verify via diagram API
    const selected = ctx.diagram.getSelectedNodes();
    expect(selected).toHaveLength(2);
    expect(selected.map((n) => n.id)).toEqual([nodeIds[0], nodeIds[2]]);
  });

  it("selects nodes in add mode", async () => {
    // First select one node
    await selectionTools.select_nodes.handler(ctx, {
      nodeIds: [nodeIds[0]],
      mode: "replace",
    });

    // Then add another
    const result = await selectionTools.select_nodes.handler(ctx, {
      nodeIds: [nodeIds[1]],
      mode: "add",
    });

    expect(result.selectedNodeIds).toHaveLength(2);
    expect(result.selectedNodeIds).toContain(nodeIds[0]);
    expect(result.selectedNodeIds).toContain(nodeIds[1]);
  });

  it("removes nodes from selection in remove mode", async () => {
    // Select all
    await selectionTools.select_nodes.handler(ctx, {
      nodeIds: nodeIds,
      mode: "replace",
    });

    // Remove one
    const result = await selectionTools.select_nodes.handler(ctx, {
      nodeIds: [nodeIds[1]],
      mode: "remove",
    });

    expect(result.selectedNodeIds).toHaveLength(2);
    expect(result.selectedNodeIds).not.toContain(nodeIds[1]);
  });

  it("clear_selection clears all selected nodes", async () => {
    // Select all nodes
    await selectionTools.select_nodes.handler(ctx, {
      nodeIds: nodeIds,
      mode: "replace",
    });

    const selectedBefore = ctx.diagram.getSelectedNodes();
    expect(selectedBefore).toHaveLength(3);

    // Clear selection
    const clearResult = await selectionTools.clear_selection.handler(ctx, {});
    expect(clearResult.cleared).toBe(true);

    const selectedAfter = ctx.diagram.getSelectedNodes();
    expect(selectedAfter).toHaveLength(0);
  });
});

describe("delete_nodes", () => {
  let ctx: ServerContext;
  let nodeIds: string[];

  beforeEach(async () => {
    ctx = createTestContext();

    // Create a chain: Input -> Linear -> ReLU -> CrossEntropyLoss
    const input = await graphTools.create_node.handler(ctx, {
      stereotype: "Input",
      position: { x: 200, y: 50 },
    });
    const lin = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 200, y: 150 },
      config: { params: { in_features: "784", out_features: "128" } },
    });
    const relu = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 200, y: 250 },
    });
    const loss = await graphTools.create_node.handler(ctx, {
      stereotype: "CrossEntropyLoss",
      position: { x: 200, y: 350 },
    });
    nodeIds = [input.nodeId, lin.nodeId, relu.nodeId, loss.nodeId];

    // Add edges between them
    ctx.diagram.addEdge(nodeIds[0], nodeIds[1]);
    ctx.diagram.addEdge(nodeIds[1], nodeIds[2]);
    ctx.diagram.addEdge(nodeIds[2], nodeIds[3]);
  });

  it("removes nodes and returns correct deletedNodeIds", async () => {
    expect(ctx.diagram.nodes).toHaveLength(4);
    expect(ctx.diagram.edges).toHaveLength(3);

    // Delete the ReLU node
    const result = await graphTools.delete_nodes.handler(ctx, {
      nodeIds: [nodeIds[2]],
    });

    expect(result.deletedNodeIds).toEqual([nodeIds[2]]);
    expect(result.deletedEdgeIds.length).toBeGreaterThanOrEqual(2); // ReLU edges removed

    // Verify diagram state
    expect(ctx.diagram.nodes).toHaveLength(3);
    expect(ctx.diagram.getNodeById(nodeIds[2])).toBeUndefined();
  });

  it("removes edges connected to deleted nodes", async () => {
    const result = await graphTools.delete_nodes.handler(ctx, {
      nodeIds: [nodeIds[1]], // Delete Linear
    });

    // Should remove edges: Input->Linear and Linear->ReLU
    expect(result.deletedEdgeIds).toHaveLength(2);
    expect(ctx.diagram.edges).toHaveLength(1); // Only ReLU->Loss remains
  });

  it("throws NodeNotFoundError for non-existent node", async () => {
    await expect(
      graphTools.delete_nodes.handler(ctx, {
        nodeIds: ["non-existent-id"],
      })
    ).rejects.toThrow(NodeNotFoundError);
  });

  it("can delete multiple nodes at once", async () => {
    const result = await graphTools.delete_nodes.handler(ctx, {
      nodeIds: [nodeIds[1], nodeIds[2]], // Delete Linear and ReLU
    });

    expect(result.deletedNodeIds).toHaveLength(2);
    expect(ctx.diagram.nodes).toHaveLength(2); // Only Input and Loss remain
  });
});

describe("connect_nodes", () => {
  let ctx: ServerContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  it("connects two nodes and returns edge ID", async () => {
    const src = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
    });
    const tgt = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 100, y: 150 },
    });

    const result = await graphTools.connect_nodes.handler(ctx, {
      source: src.nodeId,
      target: tgt.nodeId,
    });

    expect(result.edgeId).toBeTruthy();
    expect(result.source).toBe(src.nodeId);
    expect(result.target).toBe(tgt.nodeId);

    // Verify edge in diagram
    const edge = ctx.diagram.edges.find((e) => e.id === result.edgeId);
    expect(edge).toBeTruthy();
    expect(edge!.source).toBe(src.nodeId);
    expect(edge!.target).toBe(tgt.nodeId);
  });

  it("throws NodeNotFoundError for bad source", async () => {
    const tgt = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 100, y: 150 },
    });

    await expect(
      graphTools.connect_nodes.handler(ctx, {
        source: "non-existent-source",
        target: tgt.nodeId,
      })
    ).rejects.toThrow(NodeNotFoundError);
  });
});

describe("move_nodes", () => {
  let ctx: ServerContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  it("moves a single node to a new position", async () => {
    const node = await graphTools.create_node.handler(ctx, {
      stereotype: "Conv2d",
      position: { x: 50, y: 50 },
    });

    await graphTools.move_nodes.handler(ctx, {
      positions: [{ id: node.nodeId, x: 200, y: 300 }],
    });

    const moved = ctx.diagram.getNodeById(node.nodeId)!;
    expect(moved.position).toEqual({ x: 200, y: 300 });
  });

  it("moves multiple nodes simultaneously", async () => {
    const a = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 0, y: 0 },
    });
    const b = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 100, y: 100 },
    });

    await graphTools.move_nodes.handler(ctx, {
      positions: [
        { id: a.nodeId, x: 500, y: 500 },
        { id: b.nodeId, x: 600, y: 600 },
      ],
    });

    expect(ctx.diagram.getNodeById(a.nodeId)!.position).toEqual({ x: 500, y: 500 });
    expect(ctx.diagram.getNodeById(b.nodeId)!.position).toEqual({ x: 600, y: 600 });
  });
});

describe("disconnect_nodes", () => {
  let ctx: ServerContext;

  beforeEach(() => {
    ctx = createTestContext();
  });

  it("disconnects two previously connected nodes", async () => {
    const a = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 100, y: 50 },
    });
    const b = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 100, y: 150 },
    });

    // Connect first
    await graphTools.connect_nodes.handler(ctx, {
      source: a.nodeId,
      target: b.nodeId,
    });
    expect(ctx.diagram.edges).toHaveLength(1);

    // Disconnect
    const result = await graphTools.disconnect_nodes.handler(ctx, {
      source: a.nodeId,
      target: b.nodeId,
    });
    expect(result.removedEdgeIds).toHaveLength(1);
    expect(ctx.diagram.edges).toHaveLength(0);
  });
});

describe("Full pipeline compilation", () => {
  it("builds MNIST classifier diagram and compiles to NNTree", async () => {
    const ctx = createTestContext();

    // Build the graph programmatically (same as MCP create_node calls)
    const inputResult = await graphTools.create_node.handler(ctx, {
      stereotype: "Input",
      position: { x: 200, y: 50 },
    });
    const lin1Result = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 200, y: 150 },
      config: { params: { in_features: "784", out_features: "128" } },
    });
    const reluResult = await graphTools.create_node.handler(ctx, {
      stereotype: "ReLU",
      position: { x: 200, y: 250 },
    });
    const lin2Result = await graphTools.create_node.handler(ctx, {
      stereotype: "Linear",
      position: { x: 200, y: 350 },
      config: { params: { in_features: "128", out_features: "10" } },
    });
    const lossResult = await graphTools.create_node.handler(ctx, {
      stereotype: "CrossEntropyLoss",
      position: { x: 200, y: 450 },
    });

    // Connect: Input -> Linear1 -> ReLU -> Linear2 -> Loss
    ctx.diagram.addEdge(inputResult.nodeId, lin1Result.nodeId);
    ctx.diagram.addEdge(lin1Result.nodeId, reluResult.nodeId);
    ctx.diagram.addEdge(reluResult.nodeId, lin2Result.nodeId);
    ctx.diagram.addEdge(lin2Result.nodeId, lossResult.nodeId);

    expect(ctx.diagram.nodes).toHaveLength(5);
    expect(ctx.diagram.edges).toHaveLength(4);

    // Compile via NNTree
    const nntree = new NNTree(ctx.diagram);
    const json = nntree.toJson();
    const parsed = JSON.parse(json);

    expect(parsed.root).toBeTruthy();
    expect(parsed.lossNode).toBeTruthy();
    expect(parsed.lossNode.stereotype).toBe("CrossEntropyLoss");
    expect(parsed.lossNode.taskType).toBe("classification");
    expect(parsed.nodes).toBeTruthy();
    expect(Object.keys(parsed.nodes).length).toBeGreaterThanOrEqual(1);
  });
});
