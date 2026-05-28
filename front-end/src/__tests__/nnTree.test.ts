import { describe, it, expect, vi, afterAll } from "vitest";
import { type Node, type Edge } from "@xyflow/svelte";
import { Diagram } from "../Diagram.svelte";
import { NNTree } from "../conversion/nnTree";
import type { SequentialData } from "../conversion/nnTree";
import { stubWindow, unstubWindow, node, edge } from "./helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** mninst.json — simple sequential chain */
function sequentialFixture() {
  // 8 non-loss nodes, 1 loss, 8 edges
  const nodes: Node[] = [
    node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
    node("fc1", "Linear", "Linear_0", { in_features: { value: "784", position: "top" }, out_features: { value: "350", position: "bottom" }, bias: { value: "True" } }, { color: "#4779c4" }),
    node("act1", "Tanh", "Tanh_0", {}, { color: "#f4a460" }),
    node("fc2", "Linear", "Linear_1", { in_features: { value: "350", position: "top" }, out_features: { value: "175", position: "bottom" }, bias: { value: "True" } }, { color: "#4779c4" }),
    node("act2", "Tanh", "Tanh_1", {}, { color: "#f4a460" }),
    node("fc3", "Linear", "Linear_2", { in_features: { value: "175", position: "top" }, out_features: { value: "50", position: "bottom" }, bias: { value: "True" } }, { color: "#4779c4" }),
    node("act3", "Tanh", "Tanh_2", {}, { color: "#f4a460" }),
    node("fc4", "Linear", "Linear_3", { in_features: { value: "50", position: "top" }, out_features: { value: "10", position: "bottom" }, bias: { value: "True" } }, { color: "#4779c4" }),
    node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", { reduction: { value: "mean" } }, { color: "#cd5c5c", isLoss: true }),
  ];
  const edges: Edge[] = [
    edge("e1", "input", "fc1"),
    edge("e2", "fc1", "act1"),
    edge("e3", "act1", "fc2"),
    edge("e4", "fc2", "act2"),
    edge("e5", "act2", "fc3"),
    edge("e6", "fc3", "act3"),
    edge("e7", "act3", "fc4"),
    edge("e8", "fc4", "loss"),
  ];
  return { nodes, edges };
}

/** mnist_skips.json — skip connections with Addition joins */
function skipFixture() {
  const nodes: Node[] = [
    node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
    node("fc1", "Linear", "Linear_0", { in_features: { value: "784", position: "top" }, out_features: { value: "350", position: "bottom" } }, { color: "#4779c4" }),
    node("act1", "Tanh", "Tanh_0", {}, { color: "#f4a460" }),
    node("fc2", "Linear", "Linear_1", { in_features: { value: "350", position: "top" }, out_features: { value: "175", position: "bottom" } }, { color: "#4779c4" }),
    node("act2", "Tanh", "Tanh_1", {}, { color: "#f4a460" }),
    node("fc3", "Linear", "Linear_2", { in_features: { value: "175", position: "top" }, out_features: { value: "50", position: "bottom" } }, { color: "#4779c4" }),
    node("act3", "Tanh", "Tanh_2", {}, { color: "#f4a460" }),
    node("fc4", "Linear", "Linear_3", { in_features: { value: "50", position: "top" }, out_features: { value: "10", position: "bottom" } }, { color: "#4779c4" }),
    node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    node("skip_fc", "Linear", "Linear_4", { in_features: { value: "784" }, out_features: { value: "784" } }, { color: "#4779c4" }),
    node("skip_act", "Tanh", "Tanh_3", {}, { color: "#f4a460" }),
    node("join0", "Addition", "Addition_0", {},
      { type: "join", color: "#888888" }),
    node("skip2_fc", "Linear", "Linear_5", { in_features: { value: "350" }, out_features: { value: "350" } }, { color: "#4779c4" }),
    node("skip2_act", "Tanh", "Tanh_4", {}, { color: "#f4a460" }),
    node("join1", "Addition", "Addition_1", {},
      { type: "join", color: "#888888" }),
  ];
  const edges: Edge[] = [
    edge("e1", "input", "skip_fc"),
    edge("e2", "skip_fc", "skip_act"),
    edge("e3", "skip_act", "join0", { targetHandle: "in-0" }),
    edge("e4", "input", "join0", { targetHandle: "in-1" }),
    edge("e5", "join0", "fc1", { sourceHandle: "out" }),
    edge("e6", "fc1", "act1"),
    edge("e7", "act1", "skip2_fc"),
    edge("e8", "skip2_fc", "skip2_act"),
    edge("e9", "skip2_act", "join1", { targetHandle: "in-1" }),
    edge("e10", "act1", "join1", { targetHandle: "in-0" }),
    edge("e11", "join1", "fc2", { sourceHandle: "out" }),
    edge("e12", "fc2", "act2"),
    edge("e13", "act2", "fc3"),
    edge("e14", "fc3", "act3"),
    edge("e15", "act3", "fc4"),
    edge("e16", "fc4", "loss"),
  ];
  return { nodes, edges };
}

/** autoencoder_mnist.json — encoder-decoder with skip Addition */
function autoencoderFixture() {
  const nodes: Node[] = [
    node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
    node("enc_fc", "Linear", "Encoder_Linear", { in_features: { value: "784", position: "top" }, out_features: { value: "128", position: "bottom" } }, { color: "#4779c4" }),
    node("enc_act", "Tanh", "Encoder_Tanh", {}, { color: "#f4a460" }),
    node("bot_fc1", "Linear", "Bottleneck_Linear_1", { in_features: { value: "128", position: "top" }, out_features: { value: "64", position: "bottom" } }, { color: "#4779c4" }),
    node("bot_act1", "Tanh", "Bottleneck_Tanh_1", {}, { color: "#f4a460" }),
    node("bot_fc2", "Linear", "Bottleneck_Linear_2", { in_features: { value: "64", position: "top" }, out_features: { value: "128", position: "bottom" } }, { color: "#4779c4" }),
    node("bot_act2", "Tanh", "Bottleneck_Tanh_2", {}, { color: "#f4a460" }),
    node("join", "Addition", "Addition", {},
      { type: "join", color: "#888888" }),
    node("out_fc", "Linear", "Output_Linear", { in_features: { value: "128", position: "top" }, out_features: { value: "784", position: "bottom" } }, { color: "#4779c4" }),
    node("out_act", "Sigmoid", "Sigmoid_0", {}, { color: "#f4a460" }),
    node("loss", "MSELoss", "MSELoss_0", { reduction: { value: "mean" } }, { color: "#cd5c5c", isLoss: true }),
  ];
  const edges: Edge[] = [
    edge("e1", "input", "enc_fc"),
    edge("e2", "enc_fc", "enc_act"),
    edge("e3", "enc_act", "bot_fc1"),
    edge("e4", "bot_fc1", "bot_act1"),
    edge("e5", "bot_act1", "bot_fc2"),
    edge("e6", "bot_fc2", "bot_act2"),
    edge("e7", "bot_act2", "join", { targetHandle: "in-0" }),
    edge("e8", "enc_act", "join", { targetHandle: "in-1" }),
    edge("e9", "join", "out_fc", { sourceHandle: "out" }),
    edge("e10", "out_fc", "out_act"),
    edge("e11", "out_act", "loss"),
  ];
  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

stubWindow();
afterAll(() => unstubWindow());

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("NNTree — sequential chain", () => {
  const { nodes, edges } = sequentialFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);
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

  it("includes all non-loss layers in correct order", () => {
    const rootNode = tree.nodes.get("input");
    expect(rootNode).toBeDefined();
    expect(rootNode!.isSequential()).toBe(true);

    const seq = rootNode!.data as SequentialData;
    const nonLossNodes = nodes.filter((n) => !n.data.isLoss);
    expect(seq.layers).toHaveLength(nonLossNodes.length);

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
    const seq = (tree.nodes.get("input")!.data as SequentialData);
    const linear0 = seq.layers.find((l) => l.name === "Linear_0");
    expect(linear0).toBeDefined();
    expect(linear0!.params.out_features.value).toBe("350");
    expect(linear0!.params.in_features.value).toBe("784");
  });

  it("serializes to valid JSON with root, lossNode, nodes", () => {
    expect(json.root).toBe("input");
    expect(json.lossNode).toBeDefined();
    expect(json.lossNode.stereotype).toBe("CrossEntropyLoss");
    expect(json.nodes).toBeDefined();
    expect(json.nodes.input).toBeDefined();
  });
});

describe("NNTree — skip connections with joins", () => {
  const { nodes, edges } = skipFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("produces multiple tree nodes", () => {
    expect(tree.nodes.size).toBeGreaterThan(1);
  });

  it("includes both Addition join nodes", () => {
    expect(tree.nodes.has("join0")).toBe(true);
    expect(tree.nodes.has("join1")).toBe(true);
  });

  it("marks join nodes with type=join and pythonClassName=ops.Addition", () => {
    const j0 = tree.nodes.get("join0");
    expect(j0).toBeDefined();
    expect(j0!.isJoin()).toBe(true);
    expect(j0!.data).toMatchObject({
      type: "join",
      stereotype: "Addition",
      pythonClassName: "ops.Addition",
    });

    const j1 = tree.nodes.get("join1");
    expect(j1).toBeDefined();
    expect(j1!.isJoin()).toBe(true);
  });

  it("preserves sequential segments between joins", () => {
    // skip_fc + skip_act form a sequential before join0
    const skipSeq = tree.nodes.get("skip_fc");
    expect(skipSeq).toBeDefined();
    expect(skipSeq!.isSequential()).toBe(true);
    const seqData = skipSeq!.data as SequentialData;
    expect(seqData.layers).toHaveLength(2);
    expect(seqData.layers[0].stereotype).toBe("Linear");
    expect(seqData.layers[1].stereotype).toBe("Tanh");
  });

  it("join nodes route to correct children", () => {
    // join0's single child is fc1 (the main chain)
    const j0 = tree.nodes.get("join0");
    expect(j0).toBeDefined();
    expect(j0!.children).toEqual(["fc1"]);

    // fc1 is a sequential that branches into skip2_fc and join1
    const fc1Node = tree.nodes.get("fc1");
    expect(fc1Node).toBeDefined();
    expect(fc1Node!.isSequential()).toBe(true);
    expect(fc1Node!.children).toEqual(["skip2_fc", "join1"]);

    // join1 routes to fc2
    const j1 = tree.nodes.get("join1");
    expect(j1).toBeDefined();
    expect(j1!.children).toEqual(["fc2"]);
  });

  it("sets lossNode correctly", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });

  it("serializes join data in JSON output", () => {
    const j = JSON.parse(tree.toJson());
    expect(j.lossNode.stereotype).toBe("CrossEntropyLoss");
    expect(Object.keys(j.nodes)).toContain("join0");
    expect(Object.keys(j.nodes)).toContain("join1");
  });
});

describe("NNTree — autoencoder with skip", () => {
  const { nodes, edges } = autoencoderFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("produces valid tree with join node", () => {
    expect(tree.nodes.has("join")).toBe(true);
    expect(tree.nodes.size).toBeGreaterThan(0);
  });

  it("encoder path layers inside input sequential", () => {
    const inSeq = tree.nodes.get("input");
    expect(inSeq).toBeDefined();
    expect(inSeq!.isSequential()).toBe(true);
    const seq = inSeq!.data as SequentialData;
    expect(seq.layers).toHaveLength(3);
    expect(seq.layers[0].stereotype).toBe("Input");
    expect(seq.layers[1].stereotype).toBe("Linear");
    expect(seq.layers[1].name).toBe("Encoder_Linear");
    expect(seq.layers[2].stereotype).toBe("Tanh");
    expect(seq.layers[2].name).toBe("Encoder_Tanh");
  });

  it("bottleneck forms sequential chain", () => {
    const botSeq = tree.nodes.get("bot_fc1");
    expect(botSeq).toBeDefined();
    expect(botSeq!.isSequential()).toBe(true);
    const seq = botSeq!.data as SequentialData;
    expect(seq.layers).toHaveLength(4);
    expect(seq.layers.map((l) => l.stereotype)).toEqual([
      "Linear", "Tanh", "Linear", "Tanh",
    ]);
  });

  it("join node receives from both encoder and bottleneck paths", () => {
    const joinNode = tree.nodes.get("join");
    expect(joinNode).toBeDefined();
    expect(joinNode!.isJoin()).toBe(true);
    expect(joinNode!.data).toMatchObject({
      type: "join",
      stereotype: "Addition",
      pythonClassName: "ops.Addition",
    });
    // join's child is the decoder
    expect(joinNode!.children).toEqual(["out_fc"]);
  });

  it("decoder path in sequential with Sigmoid activation", () => {
    const outSeq = tree.nodes.get("out_fc");
    expect(outSeq).toBeDefined();
    expect(outSeq!.isSequential()).toBe(true);
    const seq = outSeq!.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].stereotype).toBe("Linear");
    expect(seq.layers[0].name).toBe("Output_Linear");
    expect(seq.layers[1].stereotype).toBe("Sigmoid");
  });

  it("sets MSELoss as lossNode with regression taskType", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("MSELoss");
    expect(tree.lossNode!.taskType).toBe("regression");
    expect(tree.lossNode!.pythonClassName).toBe("nn.MSELoss");
  });
});

describe("NNTree — error handling", () => {
  it("throws on empty diagram (no Input node)", () => {
    const d = new Diagram();
    d.nodes = [];
    d.edges = [];
    expect(() => new NNTree(d)).toThrow("Expected exactly one input node");
  });

  it("throws on multiple Input nodes", () => {
    const d = new Diagram();
    d.nodes = [
      node("i1", "Input", "Input_0", {}, { isInput: true }),
      node("i2", "Input", "Input_1", {}, { isInput: true }),
    ];
    d.edges = [];
    expect(() => new NNTree(d)).toThrow("Expected exactly one input node");
  });

  it("throws when node stereotypes don't include Input", () => {
    const d = new Diagram();
    d.nodes = [node("r1", "ReLU", "ReLU_0")];
    d.edges = [];
    expect(() => new NNTree(d)).toThrow("Expected exactly one input node");
  });

  it("warns on graph cycle but still produces a tree", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const d = new Diagram();
    d.nodes = [
      node("i", "Input", "Input_0", { out_features: { value: "10" } }, { isInput: true }),
      node("a", "Linear", "Linear_0"),
      node("b", "Tanh", "Tanh_0"),
    ];
    // Cycle: Input → A → B → A
    d.edges = [
      edge("e1", "i", "a"),
      edge("e2", "a", "b"),
      edge("e3", "b", "a"),
    ];
    const tree = new NNTree(d);
    // Tree still produces output despite cycle
    expect(tree.nodes.size).toBeGreaterThan(0);
    expect(tree.root).toBeDefined();
    // Warn about loop detected
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining("is visited, there is a loop"),
    );
    warnSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Subflow compilation tests
// ---------------------------------------------------------------------------

describe("NNTree — subflow boundary mapping", () => {
  function subflowFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("sub1", "", "Sub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("lin1", "Linear", "Linear_0", { out_features: { value: "10" } }, { parentId: "sub1", color: "#4779c4" }),
      node("tan1", "Tanh", "Tanh_0", {}, { parentId: "sub1", color: "#f4a460" }),
      node("lin2", "Linear", "Linear_1", { out_features: { value: "5" } }, { parentId: "sub1", color: "#4779c4" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
      edge("e3", "lin1", "tan1"),
      edge("e4", "tan1", "lin2"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = subflowFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("includes subflow node as a tree node", () => {
    expect(tree.nodes.has("sub1")).toBe(true);
  });

  it("compiles subflow internals as SequentialData with ordered layers", () => {
    const sub1 = tree.nodes.get("sub1")!;
    expect(sub1.isSequential()).toBe(true);
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(3);
    expect(seq.layers[0]).toMatchObject({ name: "Linear_0", stereotype: "Linear", pythonClassName: "nn.Linear" });
    expect(seq.layers[1]).toMatchObject({ name: "Tanh_0", stereotype: "Tanh", pythonClassName: "nn.Tanh" });
    expect(seq.layers[2]).toMatchObject({ name: "Linear_1", stereotype: "Linear", pythonClassName: "nn.Linear" });
  });

  it("routes Input sequential to subflow tree node", () => {
    const input = tree.nodes.get("input")!;
    expect(input.children).toEqual(["sub1"]);
  });

  it("subflow has no tree children (loss absorbed as lossNode)", () => {
    const sub1 = tree.nodes.get("sub1")!;
    expect(sub1.children).toEqual([]);
  });

  it("lossNode is CrossEntropyLoss", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });
});

describe("NNTree — Repeat stereotype unrolling", () => {
  function repeatFixture(iterations: string) {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("sub1", "Repeat", "Sub_0", { iterations: { value: iterations } }, { type: "subflow", color: "#9b59b6" }),
      node("conv1", "Conv2d", "Conv2d_0", { in_channels: { value: "1" }, out_channels: { value: "16" } }, { parentId: "sub1", color: "#20b2aa" }),
      node("tan1", "Tanh", "Tanh_0", {}, { parentId: "sub1", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
      edge("e3", "conv1", "tan1"),
    ];
    return { nodes, edges };
  }

  it("unrolls 2 internal nodes × 3 iterations = 6 layers", () => {
    const { nodes, edges } = repeatFixture("3");
    const d = new Diagram();
    d.nodes = nodes;
    d.edges = edges;
    const tree = new NNTree(d);
    const sub1 = tree.nodes.get("sub1")!;
    expect(sub1.isSequential()).toBe(true);
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(6);
    const expected = ["Conv2d", "Tanh", "Conv2d", "Tanh", "Conv2d", "Tanh"];
    seq.layers.forEach((layer, i) => {
      expect(layer.stereotype).toBe(expected[i]);
      expect(layer.type).toBe("module");
    });
  });

  it("iterations=1 produces single copy (no unrolling)", () => {
    const { nodes, edges } = repeatFixture("1");
    const d = new Diagram();
    d.nodes = nodes;
    d.edges = edges;
    const tree = new NNTree(d);
    const sub1 = tree.nodes.get("sub1")!;
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].stereotype).toBe("Conv2d");
    expect(seq.layers[1].stereotype).toBe("Tanh");
  });
});

describe("NNTree — subflow in sequential chain", () => {
  function chainFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("linA", "Linear", "Linear_0", { out_features: { value: "10" } }, { color: "#4779c4" }),
      node("sub1", "", "Sub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("linB", "Linear", "Linear_1", { out_features: { value: "5" } }, { parentId: "sub1", color: "#4779c4" }),
      node("tanC", "Tanh", "Tanh_2", {}, { parentId: "sub1", color: "#f4a460" }),
      node("linD", "Linear", "Linear_3", { out_features: { value: "3" } }, { color: "#4779c4" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "linA"),
      edge("e2", "linA", "sub1"),
      edge("e3", "sub1", "linD"),
      edge("e4", "linD", "loss"),
      edge("e5", "linB", "tanC"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = chainFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("creates three tree nodes (input, sub1, linD)", () => {
    expect(tree.nodes.has("input")).toBe(true);
    expect(tree.nodes.has("sub1")).toBe(true);
    expect(tree.nodes.has("linD")).toBe(true);
    expect(tree.nodes.size).toBe(3);
  });

  it("Input sequential contains Input + Linear_A", () => {
    const input = tree.nodes.get("input")!;
    expect(input.isSequential()).toBe(true);
    const seq = input.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].stereotype).toBe("Input");
    expect(seq.layers[1].stereotype).toBe("Linear");
    expect(seq.layers[1].name).toBe("Linear_0");
    expect(input.children).toEqual(["sub1"]);
  });

  it("subflow compiles internals and routes to linD", () => {
    const sub1 = tree.nodes.get("sub1")!;
    expect(sub1.isSequential()).toBe(true);
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].name).toBe("Linear_1");
    expect(seq.layers[1].name).toBe("Tanh_2");
    expect(sub1.children).toEqual(["linD"]);
  });

  it("linD is leaf sequential with loss absorbed", () => {
    const linD = tree.nodes.get("linD")!;
    expect(linD.isSequential()).toBe(true);
    const seq = linD.data as SequentialData;
    expect(seq.layers).toHaveLength(1);
    expect(seq.layers[0].stereotype).toBe("Linear");
    expect(seq.layers[0].name).toBe("Linear_3");
    expect(linD.children).toEqual([]);
  });

  it("sets lossNode correctly", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });
});

describe("NNTree — subflow edge cases", () => {
  it("warns on empty subflow (no internal nodes)", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const d = new Diagram();
    d.nodes = [
      node("input", "Input", "Input_0", {}, { isInput: true }),
      node("sub1", "", "Sub_0", {}, { type: "subflow" }),
      node("loss", "CrossEntropyLoss", "Loss", {}, { isLoss: true }),
    ];
    d.edges = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
    ];
    const tree = new NNTree(d);
    expect(tree.nodes.has("sub1")).toBe(true);
    const sub1 = tree.nodes.get("sub1")!;
    expect(sub1.isSequential()).toBe(true);
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(0);
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("no internal nodes"));
    warnSpy.mockRestore();
  });

  it("throws on cycle inside subflow internal graph", () => {
    const d = new Diagram();
    d.nodes = [
      node("input", "Input", "Input_0", {}, { isInput: true }),
      node("sub1", "", "Sub_0", {}, { type: "subflow" }),
      node("a", "Linear", "A", {}, { parentId: "sub1" }),
      node("b", "Tanh", "B", {}, { parentId: "sub1" }),
      node("c", "ReLU", "C", {}, { parentId: "sub1" }),
      node("loss", "CrossEntropyLoss", "Loss", {}, { isLoss: true }),
    ];
    d.edges = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
      edge("ea", "a", "b"),
      edge("eb", "b", "c"),
      edge("ec", "c", "a"),
    ];
    expect(() => new NNTree(d)).toThrow("cycle");
  });

  it("compiles subflow with non-SubFlow stereotype (no unrolling)", () => {
    const d = new Diagram();
    d.nodes = [
      node("input", "Input", "Input_0", {}, { isInput: true }),
      node("sub1", "Linear", "Sub_0", {}, { type: "subflow" }),
      node("a", "Tanh", "Tanh_0", {}, { parentId: "sub1" }),
      node("loss", "CrossEntropyLoss", "Loss", {}, { isLoss: true }),
    ];
    d.edges = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
    ];
    const tree = new NNTree(d);
    expect(tree.nodes.has("sub1")).toBe(true);
    const sub1 = tree.nodes.get("sub1")!;
    const seq = sub1.data as SequentialData;
    expect(seq.layers).toHaveLength(1);
    expect(seq.layers[0].stereotype).toBe("Tanh");
  });
});

// ---------------------------------------------------------------------------
// Nested subflow compilation tests
// ---------------------------------------------------------------------------

describe("NNTree — basic nested subflow", () => {
  function nestedSubflowFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("outer", "", "OuterSub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("inner", "", "InnerSub_0", {}, { type: "subflow", parentId: "outer", color: "#8e44ad" }),
      node("lin", "Linear", "Linear_0", { out_features: { value: "10" } }, { parentId: "inner", color: "#4779c4" }),
      node("tan", "Tanh", "Tanh_0", {}, { parentId: "inner", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "outer"),
      edge("e2", "outer", "loss"),
      edge("e3", "lin", "tan"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = nestedSubflowFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("compiles nested subflow as single tree node", () => {
    expect(tree.nodes.has("outer")).toBe(true);
    expect(tree.nodes.size).toBe(2); // input + outer
  });

  it("flattens inner subflow layers into outer sequential data", () => {
    const outer = tree.nodes.get("outer")!;
    expect(outer.isSequential()).toBe(true);
    const seq = outer.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0]).toMatchObject({ name: "Linear_0", stereotype: "Linear", pythonClassName: "nn.Linear" });
    expect(seq.layers[1]).toMatchObject({ name: "Tanh_0", stereotype: "Tanh", pythonClassName: "nn.Tanh" });
  });

  it("routes Input sequential to outer subflow", () => {
    const input = tree.nodes.get("input")!;
    expect(input.children).toEqual(["outer"]);
  });

  it("outer subflow has no tree children (loss absorbed)", () => {
    const outer = tree.nodes.get("outer")!;
    expect(outer.children).toEqual([]);
  });

  it("sets lossNode correctly", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });
});

describe("NNTree — nested subflow with sibling nodes", () => {
  function nestedWithSiblingFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("outer", "", "OuterSub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("inner", "", "InnerSub_0", {}, { type: "subflow", parentId: "outer", color: "#8e44ad" }),
      node("lin", "Linear", "Linear_0", { out_features: { value: "10" } }, { parentId: "inner", color: "#4779c4" }),
      node("tan", "Tanh", "Tanh_0", {}, { parentId: "inner", color: "#f4a460" }),
      node("relu", "ReLU", "ReLU_0", {}, { parentId: "outer", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "outer"),
      edge("e2", "outer", "loss"),
      edge("e3", "lin", "tan"),
      edge("e4", "inner", "relu"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = nestedWithSiblingFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("compiles all layers in topological order", () => {
    const outer = tree.nodes.get("outer")!;
    const seq = outer.data as SequentialData;
    expect(seq.layers).toHaveLength(3);
    expect(seq.layers[0].stereotype).toBe("Linear");
    expect(seq.layers[1].stereotype).toBe("Tanh");
    expect(seq.layers[2].stereotype).toBe("ReLU");
  });

  it("inner subflow does not appear as separate tree node", () => {
    expect(tree.nodes.has("inner")).toBe(false);
  });
});

describe("NNTree — nested Repeat subflow", () => {
  function nestedRepeatFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("outer", "", "OuterSub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("repeat_sub", "Repeat", "Repeat_0", { iterations: { value: "3" } }, { type: "subflow", parentId: "outer", color: "#8e44ad" }),
      node("conv", "Conv2d", "Conv2d_0", { in_channels: { value: "1" }, out_channels: { value: "16" } }, { parentId: "repeat_sub", color: "#20b2aa" }),
      node("tan", "Tanh", "Tanh_0", {}, { parentId: "repeat_sub", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "outer"),
      edge("e2", "outer", "loss"),
      edge("e3", "conv", "tan"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = nestedRepeatFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("unrolls inner Repeat subflow inside outer subflow", () => {
    const outer = tree.nodes.get("outer")!;
    const seq = outer.data as SequentialData;
    expect(seq.layers).toHaveLength(6);
    const expected = ["Conv2d", "Tanh", "Conv2d", "Tanh", "Conv2d", "Tanh"];
    seq.layers.forEach((layer, i) => {
      expect(layer.stereotype).toBe(expected[i]);
      expect(layer.type).toBe("module");
    });
  });

  it("inner repeat subflow not in tree nodes", () => {
    expect(tree.nodes.has("repeat_sub")).toBe(false);
  });
});

describe("NNTree — deeply nested subflow (3 levels)", () => {
  function deepNestedFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("l1", "", "L1_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("l2", "", "L2_0", {}, { type: "subflow", parentId: "l1", color: "#8e44ad" }),
      node("l3", "", "L3_0", {}, { type: "subflow", parentId: "l2", color: "#71368a" }),
      node("lin", "Linear", "Linear_0", { out_features: { value: "10" } }, { parentId: "l3", color: "#4779c4" }),
      node("tan", "Tanh", "Tanh_0", {}, { parentId: "l3", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "l1"),
      edge("e2", "l1", "loss"),
      edge("e3", "lin", "tan"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = deepNestedFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("flattens 3 levels of nesting into single layer list", () => {
    const l1 = tree.nodes.get("l1")!;
    const seq = l1.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].stereotype).toBe("Linear");
    expect(seq.layers[1].stereotype).toBe("Tanh");
  });

  it("intermediate subflows not in tree nodes", () => {
    expect(tree.nodes.has("l2")).toBe(false);
    expect(tree.nodes.has("l3")).toBe(false);
  });
});

describe("NNTree — nested subflow in top-level chain", () => {
  function nestedInChainFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("linA", "Linear", "Linear_A", { out_features: { value: "10" } }, { color: "#4779c4" }),
      node("outer", "", "OuterSub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("inner", "", "InnerSub_0", {}, { type: "subflow", parentId: "outer", color: "#8e44ad" }),
      node("lin_inner", "Linear", "Linear_Inner", { out_features: { value: "5" } }, { parentId: "inner", color: "#4779c4" }),
      node("tan_inner", "Tanh", "Tanh_Inner", {}, { parentId: "inner", color: "#f4a460" }),
      node("linB", "Linear", "Linear_B", { out_features: { value: "3" } }, { color: "#4779c4" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "linA"),
      edge("e2", "linA", "outer"),
      edge("e3", "outer", "linB"),
      edge("e4", "linB", "loss"),
      edge("e5", "lin_inner", "tan_inner"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = nestedInChainFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);

  it("creates three tree nodes (input, outer, linB)", () => {
    expect(tree.nodes.has("input")).toBe(true);
    expect(tree.nodes.has("outer")).toBe(true);
    expect(tree.nodes.has("linB")).toBe(true);
    expect(tree.nodes.size).toBe(3);
  });

  it("Input sequential contains Input + Linear_A", () => {
    const input = tree.nodes.get("input")!;
    expect(input.isSequential()).toBe(true);
    const seq = input.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].stereotype).toBe("Input");
    expect(seq.layers[1].stereotype).toBe("Linear");
    expect(seq.layers[1].name).toBe("Linear_A");
    expect(input.children).toEqual(["outer"]);
  });

  it("outer subflow compiles inner nested layers", () => {
    const outer = tree.nodes.get("outer")!;
    expect(outer.isSequential()).toBe(true);
    const seq = outer.data as SequentialData;
    expect(seq.layers).toHaveLength(2);
    expect(seq.layers[0].name).toBe("Linear_Inner");
    expect(seq.layers[1].name).toBe("Tanh_Inner");
    expect(outer.children).toEqual(["linB"]);
  });

  it("linB is leaf sequential with loss absorbed", () => {
    const linB = tree.nodes.get("linB")!;
    expect(linB.isSequential()).toBe(true);
    const seq = linB.data as SequentialData;
    expect(seq.layers).toHaveLength(1);
    expect(seq.layers[0].stereotype).toBe("Linear");
    expect(seq.layers[0].name).toBe("Linear_B");
    expect(linB.children).toEqual([]);
  });

  it("sets lossNode correctly", () => {
    expect(tree.lossNode).not.toBeNull();
    expect(tree.lossNode!.stereotype).toBe("CrossEntropyLoss");
  });
});

// ---------------------------------------------------------------------------
// FUTURE: subflow should be type "subflow" with entryNode + internal nodes
// These tests FAIL with current code — they define the TARGET behavior
// ---------------------------------------------------------------------------

describe("NNTree — subflow should be type subflow (TARGET)", () => {
  function targetSubflowFixture() {
    const nodes: Node[] = [
      node("input", "Input", "Input_0", { out_features: { value: "784" } }, { isInput: true, color: "#27b376" }),
      node("sub1", "", "Sub_0", {}, { type: "subflow", color: "#9b59b6" }),
      node("lin", "Linear", "Linear_0", { out_features: { value: "10" } }, { parentId: "sub1", color: "#4779c4" }),
      node("tan", "Tanh", "Tanh_0", {}, { parentId: "sub1", color: "#f4a460" }),
      node("loss", "CrossEntropyLoss", "CrossEntropyLoss_0", {}, { color: "#cd5c5c", isLoss: true }),
    ];
    const edges: Edge[] = [
      edge("e1", "input", "sub1"),
      edge("e2", "sub1", "loss"),
      edge("e3", "lin", "tan"),
    ];
    return { nodes, edges };
  }

  const { nodes, edges } = targetSubflowFixture();
  const d = new Diagram();
  d.nodes = nodes;
  d.edges = edges;
  const tree = new NNTree(d);
  const sub1 = tree.nodes.get("sub1")!;

  it("subflow data type is 'subflow' not 'sequential'", () => {
    expect(sub1.data.type).toBe("subflow");
  });

  it("subflow has entryNode pointing to first internal node", () => {
    const data = sub1.data as any;
    expect(data.entryNode).toBeDefined();
    expect(typeof data.entryNode).toBe("string");
  });

  it("subflow stores internal nodes in 'nodes' map", () => {
    const data = sub1.data as any;
    expect(data.nodes).toBeDefined();
    expect(typeof data.nodes).toBe("object");
    // Should contain both internal nodes
    expect(Object.keys(data.nodes).length).toBeGreaterThanOrEqual(2);
    expect(data.nodes).toHaveProperty("lin");
    expect(data.nodes).toHaveProperty("tan");
  });

  it("internal nodes have children preserving internal topology", () => {
    const data = sub1.data as any;
    expect(data.nodes.lin.children).toContain("tan");
    expect(data.nodes.tan.children).toEqual([]); // last node, no internal children
  });

  it("serialized JSON preserves subflow structure", () => {
    const json = JSON.parse(tree.toJson());
    const subflowNode = json.nodes.sub1;
    expect(subflowNode.data.type).toBe("subflow");
    expect(subflowNode.data.entryNode).toBeDefined();
    expect(subflowNode.data.nodes).toBeDefined();
    expect(subflowNode.data.nodes.lin).toBeDefined();
  });
});
