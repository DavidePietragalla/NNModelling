import { describe, it, expect, vi } from "vitest";
import { type Node, type Edge } from "@xyflow/svelte";
import { NNTree } from "../conversion/nnTree";
import { TestDiagram, buildStereotype, param } from "./helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** mninst.json — simple sequential chain */
function sequentialFixture() {
  const nodes: Node[] = [
    {
      id: "input",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Input", name: "Input_0", color: "#27b376", params: { out_features: param("784") }, isInput: true, isLoss: false },
    },
    {
      id: "fc1",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Linear", name: "Linear_0", color: "#4779c4", params: { in_features: param("784", "top"), out_features: param("350", "bottom"), bias: param("True") }, isInput: false, isLoss: false },
    },
    {
      id: "act1",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Tanh", name: "Tanh_0", color: "#f4a460", params: {}, isInput: false, isLoss: false },
    },
    {
      id: "fc2",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Linear", name: "Linear_1", color: "#4779c4", params: { in_features: param("350", "top"), out_features: param("175", "bottom"), bias: param("True") }, isInput: false, isLoss: false },
    },
    {
      id: "act2",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Tanh", name: "Tanh_1", color: "#f4a460", params: {}, isInput: false, isLoss: false },
    },
    {
      id: "fc3",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Linear", name: "Linear_2", color: "#4779c4", params: { in_features: param("175", "top"), out_features: param("50", "bottom"), bias: param("True") }, isInput: false, isLoss: false },
    },
    {
      id: "act3",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Tanh", name: "Tanh_2", color: "#f4a460", params: {}, isInput: false, isLoss: false },
    },
    {
      id: "fc4",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "Linear", name: "Linear_3", color: "#4779c4", params: { in_features: param("50", "top"), out_features: param("10", "bottom"), bias: param("True") }, isInput: false, isLoss: false },
    },
    {
      id: "loss",
      type: "custom",
      position: { x: 0, y: 0 },
      data: { stereotype: "CrossEntropyLoss", name: "CrossEntropyLoss_0", color: "#cd5c5c", params: { reduction: param("mean") }, isInput: false, isLoss: true },
    },
  ];

  const edges: Edge[] = [
    { id: "e1", source: "input", target: "fc1" },
    { id: "e2", source: "fc1", target: "act1" },
    { id: "e3", source: "act1", target: "fc2" },
    { id: "e4", source: "fc2", target: "act2" },
    { id: "e5", source: "act2", target: "fc3" },
    { id: "e6", source: "fc3", target: "act3" },
    { id: "e7", source: "act3", target: "fc4" },
    { id: "e8", source: "fc4", target: "loss" },
  ];

  const stereotypes = [
    buildStereotype("Input", { category: "Input", params: { out_features: { type: "int", default: "784" } } }),
    buildStereotype("Linear", { params: { in_features: { type: "int", default: "Undefined", position: "top" }, out_features: { type: "int", default: "Undefined", position: "bottom" }, bias: { type: "bool", default: "True" } } }),
    buildStereotype("Tanh", { params: {} }),
    buildStereotype("CrossEntropyLoss", { category: "CrossEntropyLoss", pythonClassName: "nn.CrossEntropyLoss", taskType: "classification" }),
  ];

  return { nodes, edges, stereotypes };
}

/** mnist_skips.json — skip connections with Addition joins */
function skipFixture() {
  const nodes: Node[] = [
    { id: "input", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Input", name: "Input_0", color: "#27b376", params: { out_features: param("784") }, isInput: true, isLoss: false } },
    { id: "fc1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_0", color: "#4779c4", params: { in_features: param("784", "top"), out_features: param("350", "bottom") }, isInput: false, isLoss: false } },
    { id: "act1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_0", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "fc2", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_1", color: "#4779c4", params: { in_features: param("350", "top"), out_features: param("175", "bottom") }, isInput: false, isLoss: false } },
    { id: "act2", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_1", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "fc3", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_2", color: "#4779c4", params: { in_features: param("175", "top"), out_features: param("50", "bottom") }, isInput: false, isLoss: false } },
    { id: "act3", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_2", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "fc4", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_3", color: "#4779c4", params: { in_features: param("50", "top"), out_features: param("10", "bottom") }, isInput: false, isLoss: false } },
    { id: "loss", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "CrossEntropyLoss", name: "CrossEntropyLoss_0", color: "#cd5c5c", params: {}, isInput: false, isLoss: true } },
    // Skip path 1: Linear_4 → Tanh_3 → Addition_0
    { id: "skip_fc", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_4", color: "#4779c4", params: { in_features: param("784"), out_features: param("784") }, isInput: false, isLoss: false } },
    { id: "skip_act", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_3", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "join0", type: "join", position: { x: 0, y: 0 }, data: { stereotype: "Addition", name: "Addition_0", inputsCount: 2, color: "#888888", params: {} } },
    // Skip path 2: Linear_5 → Tanh_4 → Addition_1
    { id: "skip2_fc", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_5", color: "#4779c4", params: { in_features: param("350"), out_features: param("350") }, isInput: false, isLoss: false } },
    { id: "skip2_act", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_4", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "join1", type: "join", position: { x: 0, y: 0 }, data: { stereotype: "Addition", name: "Addition_1", inputsCount: 2, color: "#888888", params: {} } },
  ];

  const edges: Edge[] = [
    // Input → Linear_4 → Tanh_3 → Addition_0 (skip path 1)
    { id: "e1", source: "input", target: "skip_fc" },
    { id: "e2", source: "skip_fc", target: "skip_act" },
    { id: "e3", source: "skip_act", target: "join0", targetHandle: "in-0" },
    // Input → Addition_0 (direct skip connection)
    { id: "e4", source: "input", target: "join0", targetHandle: "in-1" },
    // Addition_0 → Linear_0 → Tanh_0 (main chain resumes)
    { id: "e5", source: "join0", sourceHandle: "out", target: "fc1" },
    { id: "e6", source: "fc1", target: "act1" },
    // Tanh_0 → Linear_5 → Tanh_4 → Addition_1 (skip path 2)
    { id: "e7", source: "act1", target: "skip2_fc" },
    { id: "e8", source: "skip2_fc", target: "skip2_act" },
    { id: "e9", source: "skip2_act", target: "join1", targetHandle: "in-1" },
    // Tanh_0 → Addition_1 (direct skip)
    { id: "e10", source: "act1", target: "join1", targetHandle: "in-0" },
    // Addition_1 → Linear_1 → Tanh_1 → Linear_2 → Tanh_2 → Linear_3 → Loss
    { id: "e11", source: "join1", sourceHandle: "out", target: "fc2" },
    { id: "e12", source: "fc2", target: "act2" },
    { id: "e13", source: "act2", target: "fc3" },
    { id: "e14", source: "fc3", target: "act3" },
    { id: "e15", source: "act3", target: "fc4" },
    { id: "e16", source: "fc4", target: "loss" },
  ];

  const stereotypes = [
    buildStereotype("Input", { category: "Input" }),
    buildStereotype("Linear", { params: { in_features: { type: "int", default: "Undefined" }, out_features: { type: "int", default: "Undefined" } } }),
    buildStereotype("Tanh", { params: {} }),
    buildStereotype("CrossEntropyLoss", { category: "CrossEntropyLoss", pythonClassName: "nn.CrossEntropyLoss", taskType: "classification" }),
    buildStereotype("Addition", { category: "Join", pythonClassName: "ops.Addition" }),
  ];

  return { nodes, edges, stereotypes };
}

/** autoencoder_mnist.json — encoder-decoder with skip Addition */
function autoencoderFixture() {
  const nodes: Node[] = [
    { id: "input", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Input", name: "Input_0", color: "#27b376", params: { out_features: param("784") }, isInput: true, isLoss: false } },
    // Encoder
    { id: "enc_fc", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Encoder_Linear", color: "#4779c4", params: { in_features: param("784", "top"), out_features: param("128", "bottom") }, isInput: false, isLoss: false } },
    { id: "enc_act", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Encoder_Tanh", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    // Bottleneck
    { id: "bot_fc1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Bottleneck_Linear_1", color: "#4779c4", params: { in_features: param("128", "top"), out_features: param("64", "bottom") }, isInput: false, isLoss: false } },
    { id: "bot_act1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Bottleneck_Tanh_1", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "bot_fc2", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Bottleneck_Linear_2", color: "#4779c4", params: { in_features: param("64", "top"), out_features: param("128", "bottom") }, isInput: false, isLoss: false } },
    { id: "bot_act2", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Bottleneck_Tanh_2", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    // Skip addition join
    { id: "join", type: "join", position: { x: 0, y: 0 }, data: { stereotype: "Addition", name: "Addition", inputsCount: 2, color: "#888888", params: {} } },
    // Decoder
    { id: "out_fc", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Output_Linear", color: "#4779c4", params: { in_features: param("128", "top"), out_features: param("784", "bottom") }, isInput: false, isLoss: false } },
    { id: "out_act", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Sigmoid", name: "Sigmoid_0", color: "#f4a460", params: {}, isInput: false, isLoss: false } },
    { id: "loss", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "MSELoss", name: "MSELoss_0", color: "#cd5c5c", params: { reduction: param("mean") }, isInput: false, isLoss: true } },
  ];

  const edges: Edge[] = [
    { id: "e1", source: "input", target: "enc_fc" },
    { id: "e2", source: "enc_fc", target: "enc_act" },
    { id: "e3", source: "enc_act", target: "bot_fc1" },
    { id: "e4", source: "bot_fc1", target: "bot_act1" },
    { id: "e5", source: "bot_act1", target: "bot_fc2" },
    { id: "e6", source: "bot_fc2", target: "bot_act2" },
    // Skip: enc_act → Addition AND bot_act2 → Addition
    { id: "e7", source: "bot_act2", target: "join", targetHandle: "in-0" },
    { id: "e8", source: "enc_act", target: "join", targetHandle: "in-1" },
    // Decoder path
    { id: "e9", source: "join", sourceHandle: "out", target: "out_fc" },
    { id: "e10", source: "out_fc", target: "out_act" },
    { id: "e11", source: "out_act", target: "loss" },
  ];

  const stereotypes = [
    buildStereotype("Input", { category: "Input" }),
    buildStereotype("Linear", { params: { in_features: { type: "int", default: "Undefined" }, out_features: { type: "int", default: "Undefined" } } }),
    buildStereotype("Tanh", { params: {} }),
    buildStereotype("Sigmoid", { pythonClassName: "nn.Sigmoid", params: {} }),
    buildStereotype("MSELoss", { category: "MSELoss", pythonClassName: "nn.MSELoss", taskType: "regression" }),
    buildStereotype("Addition", { category: "Join", pythonClassName: "ops.Addition" }),
  ];

  return { nodes, edges, stereotypes };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("NNTree — sequential chain", () => {
  const { nodes, edges, stereotypes } = sequentialFixture();
  const diagram = new TestDiagram(nodes, edges, stereotypes);
  const tree = new NNTree(diagram as any);
  const json = JSON.parse(tree.toJson());

  it("compresses chain into single sequential node", () => {
    expect(tree.nodes.size).toBe(1);
  });

  it("uses Input node id as root", () => {
    expect(tree.root).toBe("input");
  });

  it("sets CrossEntropyLoss as lossNode with classification taskType", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
    expect(tree.lossNode!.name).toBe("CrossEntropyLoss_0");
    expect(tree.lossNode!.taskType).toBe("classification");
    expect(tree.lossNode!.pythonClassName).toBe("nn.CrossEntropyLoss");
  });

  it("includes all layers in correct order", () => {
    const rootNode = tree.nodes.get("input")!;
    expect(rootNode.isSequential()).toBe(true);
    const seq = rootNode.data as import("../conversion/nnTree").SequentialData;
    expect(seq.layers).toHaveLength(8);

    const expected = [
      { name: "Input_0", stereotype: "Input", py: "None" },
      { name: "Linear_0", stereotype: "Linear", py: "nn.Linear" },
      { name: "Tanh_0", stereotype: "Tanh", py: "nn.Tanh" },
      { name: "Linear_1", stereotype: "Linear", py: "nn.Linear" },
      { name: "Tanh_1", stereotype: "Tanh", py: "nn.Tanh" },
      { name: "Linear_2", stereotype: "Linear", py: "nn.Linear" },
      { name: "Tanh_2", stereotype: "Tanh", py: "nn.Tanh" },
      { name: "Linear_3", stereotype: "Linear", py: "nn.Linear" },
    ];

    seq.layers.forEach((layer, i) => {
      expect(layer.name).toBe(expected[i].name);
      expect(layer.stereotype).toBe(expected[i].stereotype);
      expect(layer.pythonClassName).toBe(expected[i].py);
    });
  });

  it("preserves params in sequential layers", () => {
    const rootNode = tree.nodes.get("input")!;
    const seq = rootNode.data as import("../conversion/nnTree").SequentialData;
    const linear0 = seq.layers.find((l) => l.name === "Linear_0")!;
    expect(linear0.params.out_features.value).toBe("350");
    expect(linear0.params.in_features.value).toBe("784");
  });

  it("serializes to valid JSON with root, lossNode, nodes", () => {
    expect(json.root).toBe("input");
    expect(json.lossNode).toBeDefined();
    expect(json.lossNode.stereotype).toBe("CrossEntropyLoss");
    expect(json.nodes).toBeDefined();
    expect(json.nodes["input"]).toBeDefined();
  });
});

describe("NNTree — skip connections with joins", () => {
  const { nodes, edges, stereotypes } = skipFixture();
  const diagram = new TestDiagram(nodes, edges, stereotypes);
  const tree = new NNTree(diagram as any);

  it("produces multiple tree nodes (not a single sequential)", () => {
    // Skip graph has branches → multiple NNTree nodes
    expect(tree.nodes.size).toBeGreaterThan(1);
  });

  it("includes Addition join nodes in the tree", () => {
    const joinKeys = Array.from(tree.nodes.keys()).filter((k) => k.startsWith("join"));
    expect(joinKeys.length).toBeGreaterThanOrEqual(2);
  });

  it("marks join nodes with type=join and ops.Addition pythonClassName", () => {
    const joinNode = tree.nodes.get("join0")!;
    expect(joinNode.isJoin()).toBe(true);
    expect(joinNode.data).toMatchObject({
      type: "join",
      stereotype: "Addition",
      pythonClassName: "ops.Addition",
    });
  });

  it("sets lossNode correctly", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });

  it("serializes join data in JSON output", () => {
    const json = JSON.parse(tree.toJson());
    expect(json.lossNode.stereotype).toBe("CrossEntropyLoss");
    // At least one join node in the output
    const joinKeys = Object.keys(json.nodes).filter((k) => k.startsWith("join"));
    expect(joinKeys.length).toBeGreaterThanOrEqual(2);
  });
});

describe("NNTree — autoencoder with skip", () => {
  const { nodes, edges, stereotypes } = autoencoderFixture();
  const diagram = new TestDiagram(nodes, edges, stereotypes);
  const tree = new NNTree(diagram as any);

  it("produces valid tree with join node", () => {
    expect(tree.nodes.size).toBeGreaterThan(0);
    expect(tree.nodes.has("join")).toBe(true);
  });

  it("sets MSELoss as lossNode with regression taskType", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("MSELoss");
    expect(tree.lossNode!.taskType).toBe("regression");
    expect(tree.lossNode!.pythonClassName).toBe("nn.MSELoss");
  });

  it("creates join node with Addition data", () => {
    const joinNode = tree.nodes.get("join")!;
    expect(joinNode.isJoin()).toBe(true);
    expect(joinNode.data).toMatchObject({
      type: "join",
      stereotype: "Addition",
      pythonClassName: "ops.Addition",
    });
  });
});

describe("NNTree — error handling", () => {
  it("throws on empty diagram (no Input node)", () => {
    const diagram = new TestDiagram([], [], [buildStereotype("Linear")]);
    expect(() => new NNTree(diagram as any)).toThrow("Expected exactly one input node");
  });

  it("throws on multiple Input nodes", () => {
    const input1: Node = { id: "i1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Input", name: "Input_0", color: "", params: {}, isInput: true, isLoss: false } };
    const input2: Node = { id: "i2", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Input", name: "Input_1", color: "", params: {}, isInput: true, isLoss: false } };
    const diagram = new TestDiagram([input1, input2], [], [buildStereotype("Input", { category: "Input" })]);
    expect(() => new NNTree(diagram as any)).toThrow("Expected exactly one input node");
  });

  it("throws when no Input node exists among nodes", () => {
    const relu: Node = { id: "r1", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "ReLU", name: "ReLU_0", color: "", params: {}, isInput: false, isLoss: false } };
    const diagram = new TestDiagram([relu], [], [buildStereotype("ReLU")]);
    expect(() => new NNTree(diagram as any)).toThrow("Expected exactly one input node");
  });

  it("warns on graph cycle but still produces a tree", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const input: Node = { id: "i", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Input", name: "Input_0", color: "", params: { out_features: param("10") }, isInput: true, isLoss: false } };
    const a: Node = { id: "a", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Linear", name: "Linear_0", color: "", params: {}, isInput: false, isLoss: false } };
    const b: Node = { id: "b", type: "custom", position: { x: 0, y: 0 }, data: { stereotype: "Tanh", name: "Tanh_0", color: "", params: {}, isInput: false, isLoss: false } };
    // Cycle: Input → A → B → A
    const edges: Edge[] = [
      { id: "e1", source: "i", target: "a" },
      { id: "e2", source: "a", target: "b" },
      { id: "e3", source: "b", target: "a" },
    ];
    const diagram = new TestDiagram([input, a, b], edges, [
      buildStereotype("Input", { category: "Input" }),
      buildStereotype("Linear"),
      buildStereotype("Tanh"),
    ]);
    const tree = new NNTree(diagram as any);
    expect(tree.nodes.size).toBeGreaterThan(0);
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("is visited, there is a loop"),
    );
    warnSpy.mockRestore();
  });
});
