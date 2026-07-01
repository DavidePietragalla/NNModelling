/**
 * MCP Resource Definitions — read-only views of the current diagram state.
 *
 * All resources follow the URI scheme: nnmodelling://<resource-type>/<path>
 * and return JSON content with application/json MIME type. Resources allow
 * the LLM to inspect the canvas without calling mutation tools.
 *
 * Resource list (from architecture §5):
 *   1. nnmodelling://diagram/current     — Full diagram state
 *   2. nnmodelling://node/{id}           — Single node by ID
 *   3. nnmodelling://edge/{id}           — Single edge by ID
 *   4. nnmodelling://nodes/list          — All node IDs
 *   5. nnmodelling://edges/list          — All edge IDs
 *   6. nnmodelling://selection           — Currently selected nodes/edges
 *   7. nnmodelling://stereotypes         — All available stereotypes
 *   8. nnmodelling://stereotype/{name}   — Single stereotype definition
 *   9. nnmodelling://validation          — Latest validation report
 *   10. nnmodelling://statistics         — Graph statistics
 *   11. nnmodelling://conversion         — Latest NNTree compilation output
 *   12. nnmodelling://conversion/status  — Latest conversion pipeline status
 *   13. nnmodelling://history            — Undo/redo stack sizes
 *   14. nnmodelling://events             — Accumulated event log
 *
 * @module resources/index
 */

import type { ServerContext } from "../server";
import type { Node, Edge } from "@nnmodelling/front-end/core/types";
import { NNTree } from "@nnmodelling/front-end/conversion/nnTree";

// ── Types ─────────────────────────────────────────────────────────────────

/** A single MCP resource definition with a read handler. */
interface ResourceDefinition {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
  read(uri: URL): Promise<{
    contents: Array<{
      uri: string;
      mimeType?: string;
      text: string;
    }>;
  }>;
}

// ── Helper Functions ──────────────────────────────────────────────────────

/** Extract the stereotype name from a node's data field. */
function getStereoName(node: Node): string | undefined {
  return (node.data as Record<string, unknown> | undefined)?.stereotype as string | undefined;
}

/**
 * Compute the longest path depth in the graph using BFS from all input nodes.
 * Returns 0 if there are no input nodes or the graph is empty.
 */
function computeMaxDepth(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (name: string) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  const adjacency = new Map<string, string[]>();
  for (const edge of graph.edges) {
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    adjacency.get(edge.source)!.push(edge.target);
  }

  const inputs = graph.nodes.filter((n) => {
    const stereoName = getStereoName(n);
    if (!stereoName) return false;
    const stereo = getStereotype(stereoName);
    return stereo?.isInput === true;
  });
  if (inputs.length === 0) return 0;

  let maxDepth = 0;
  for (const input of inputs) {
    const visited = new Set<string>();
    const queue: Array<{ id: string; depth: number }> = [{ id: input.id, depth: 0 }];
    visited.add(input.id);

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      maxDepth = Math.max(maxDepth, depth);
      for (const childId of adjacency.get(id) ?? []) {
        if (!visited.has(childId)) {
          visited.add(childId);
          queue.push({ id: childId, depth: depth + 1 });
        }
      }
    }
  }

  return maxDepth;
}

/**
 * Compute the average number of outgoing edges per non-loss node.
 * Loss/output nodes are excluded since they typically have no outgoing edges.
 */
function computeAvgFanOut(
  graph: { nodes: Node[]; edges: Edge[] },
  getStereotype: (name: string) => { isInput?: boolean; isLoss?: boolean } | undefined,
): number {
  let totalOut = 0;
  let count = 0;

  for (const node of graph.nodes) {
    const stereoName = getStereoName(node);
    if (stereoName) {
      const stereo = getStereotype(stereoName);
      if (stereo?.isLoss) continue;
    }

    totalOut += graph.edges.filter((e) => e.source === node.id).length;
    count++;
  }

  return count > 0 ? Math.round((totalOut / count) * 100) / 100 : 0;
}

/**
 * Detect whether the graph contains any cycles using Kahn's algorithm.
 * Returns true if the graph is acyclic (DAG).
 */
function isCycleFree(graph: { nodes: Node[]; edges: Edge[] }): boolean {
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const node of graph.nodes) {
    inDegree.set(node.id, 0);
    adjacency.set(node.id, []);
  }

  for (const edge of graph.edges) {
    adjacency.get(edge.source)?.push(edge.target);
    inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  let visitedCount = 0;
  while (queue.length > 0) {
    const id = queue.shift()!;
    visitedCount++;
    for (const neighbor of adjacency.get(id) ?? []) {
      const newDeg = (inDegree.get(neighbor) ?? 1) - 1;
      inDegree.set(neighbor, newDeg);
      if (newDeg === 0) queue.push(neighbor);
    }
  }

  return visitedCount === graph.nodes.length;
}

// ── Resource Factory ──────────────────────────────────────────────────────

/**
 * Define all MCP resources for the NNModelling server.
 *
 * Each resource is a read-only view of the current diagram state. The `read`
 * handler receives the parsed `URL` from the incoming MCP `ReadResource` request.
 */
export function defineResources(ctx: ServerContext): ResourceDefinition[] {
  return [
    // ── nnmodelling://diagram/current ────────────────────────────
    {
      uri: "nnmodelling://diagram/current",
      name: "Current Diagram",
      description: "Full diagram state including all nodes, edges, and computed statistics",
      mimeType: "application/json",
      async read() {
        const { nodes, edges } = ctx.diagram;
        const getStereo = (name: string) => ctx.diagram.getStereotype(name);

        return {
          contents: [
            {
              uri: "nnmodelling://diagram/current",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  nodes,
                  edges,
                  statistics: {
                    nodeCount: nodes.length,
                    edgeCount: edges.length,
                    moduleCount: nodes.filter((n) => n.type !== "join" && n.type !== "subflow").length,
                    joinCount: nodes.filter((n) => n.type === "join").length,
                    subflowCount: nodes.filter((n) => n.type === "subflow").length,
                    inputCount: nodes.filter((n) => {
                      const s = getStereoName(n);
                      return s ? getStereo(s)?.isInput === true : false;
                    }).length,
                    lossCount: nodes.filter((n) => {
                      const s = getStereoName(n);
                      return s ? getStereo(s)?.isLoss === true : false;
                    }).length,
                    maxDepth: computeMaxDepth(ctx.diagram, getStereo),
                    avgFanOut: computeAvgFanOut(ctx.diagram, getStereo),
                    cycleFree: isCycleFree(ctx.diagram),
                  },
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://node/{id} ──────────────────────────────────
    {
      uri: "nnmodelling://node/{id}",
      name: "Node by ID",
      description: "Get a single node by its ID, including incoming and outgoing edge lists",
      mimeType: "application/json",
      async read(uri: URL) {
        const nodeId = uri.pathname.split("/").pop()!;
        const node = ctx.diagram.getNodeById(nodeId);
        if (!node) {
          return {
            contents: [
              {
                uri: uri.href,
                mimeType: "application/json",
                text: JSON.stringify({ error: `Node '${nodeId}' not found` }, null, 2),
              },
            ],
          };
        }

        const incoming = ctx.diagram.edges.filter((e) => e.target === nodeId);
        const outgoing = ctx.diagram.edges.filter((e) => e.source === nodeId);

        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  ...node,
                  incoming,
                  outgoing,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://edge/{id} ──────────────────────────────────
    {
      uri: "nnmodelling://edge/{id}",
      name: "Edge by ID",
      description: "Get a single edge by its ID",
      mimeType: "application/json",
      async read(uri: URL) {
        const edgeId = uri.pathname.split("/").pop()!;
        const edge = ctx.diagram.edges.find((e) => e.id === edgeId);
        if (!edge) {
          return {
            contents: [
              {
                uri: uri.href,
                mimeType: "application/json",
                text: JSON.stringify({ error: `Edge '${edgeId}' not found` }, null, 2),
              },
            ],
          };
        }

        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(edge, null, 2),
            },
          ],
        };
      },
    },

    // ── nnmodelling://nodes/list ─────────────────────────────────
    {
      uri: "nnmodelling://nodes/list",
      name: "All Node IDs",
      description: "List all node IDs in the current diagram with their names and types",
      mimeType: "application/json",
      async read() {
        const summary = ctx.diagram.nodes.map((n) => ({
          id: n.id,
          name: (n.data as Record<string, unknown> | undefined)?.name ?? null,
          type: n.type ?? "custom",
          stereotype: getStereoName(n) ?? null,
        }));

        return {
          contents: [
            {
              uri: "nnmodelling://nodes/list",
              mimeType: "application/json",
              text: JSON.stringify({ nodes: summary, count: summary.length }, null, 2),
            },
          ],
        };
      },
    },

    // ── nnmodelling://edges/list ─────────────────────────────────
    {
      uri: "nnmodelling://edges/list",
      name: "All Edge IDs",
      description: "List all edge IDs in the current diagram with their source and target",
      mimeType: "application/json",
      async read() {
        const summary = ctx.diagram.edges.map((e) => ({
          id: e.id,
          source: e.source,
          target: e.target,
          sourceHandle: e.sourceHandle ?? null,
          targetHandle: e.targetHandle ?? null,
        }));

        return {
          contents: [
            {
              uri: "nnmodelling://edges/list",
              mimeType: "application/json",
              text: JSON.stringify({ edges: summary, count: summary.length }, null, 2),
            },
          ],
        };
      },
    },

    // ── nnmodelling://selection ──────────────────────────────────
    {
      uri: "nnmodelling://selection",
      name: "Current Selection",
      description: "Currently selected nodes and edges on the canvas",
      mimeType: "application/json",
      async read() {
        const selectedNodes = ctx.diagram.nodes.filter((n) => n.selected);
        const selectedEdges = ctx.diagram.edges.filter((e) => e.selected);

        return {
          contents: [
            {
              uri: "nnmodelling://selection",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  nodeIds: selectedNodes.map((n) => n.id),
                  edgeIds: selectedEdges.map((e) => e.id),
                  nodeCount: selectedNodes.length,
                  edgeCount: selectedEdges.length,
                  nodes: selectedNodes.map((n) => ({
                    id: n.id,
                    type: n.type ?? "custom",
                    name: (n.data as Record<string, unknown> | undefined)?.name ?? null,
                    stereotype: getStereoName(n) ?? null,
                    position: n.position,
                  })),
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://stereotypes ────────────────────────────────
    {
      uri: "nnmodelling://stereotypes",
      name: "All Stereotypes",
      description: "List all available stereotypes with their categories and parameter schemas",
      mimeType: "application/json",
      async read() {
        const allStereotypes = ctx.diagram.stereotypes ?? [];

        return {
          contents: [
            {
              uri: "nnmodelling://stereotypes",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  stereotypes: allStereotypes.map((s) => ({
                    name: s.name,
                    category: s.category,
                    pythonClassName: s.pythonClassName,
                    taskType: s.taskType,
                    isJoin: s.isJoin,
                    isInput: s.isInput,
                    isLoss: s.isLoss,
                    isSubFlow: s.isSubFlow,
                    parameters: s.parameters,
                    view: s.view,
                  })),
                  count: allStereotypes.length,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://stereotype/{name} ──────────────────────────
    {
      uri: "nnmodelling://stereotype/{name}",
      name: "Stereotype by Name",
      description: "Get a single stereotype definition by its name",
      mimeType: "application/json",
      async read(uri: URL) {
        const stereoName = uri.pathname.split("/").pop()!;
        const stereo = ctx.diagram.getStereotype(stereoName);

        if (!stereo) {
          return {
            contents: [
              {
                uri: uri.href,
                mimeType: "application/json",
                text: JSON.stringify(
                  { error: `Stereotype '${stereoName}' not found` },
                  null,
                  2,
                ),
              },
            ],
          };
        }

        return {
          contents: [
            {
              uri: uri.href,
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  name: stereo.name,
                  category: stereo.category,
                  pythonClassName: stereo.pythonClassName,
                  taskType: stereo.taskType,
                  expr: stereo.expr,
                  isJoin: stereo.isJoin,
                  isInput: stereo.isInput,
                  isLoss: stereo.isLoss,
                  isSubFlow: stereo.isSubFlow,
                  parameters: stereo.parameters,
                  view: stereo.view,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://validation ─────────────────────────────────
    {
      uri: "nnmodelling://validation",
      name: "Validation Report",
      description:
        "Run validation on the current diagram and return errors and warnings. " +
        "Checks: Input node count, loss node presence, orphan nodes, cycles.",
      mimeType: "application/json",
      async read() {
        const errors: Array<{
          code: string;
          message: string;
          nodeId?: string;
          edgeId?: string;
        }> = [];
        const warnings: Array<{
          code: string;
          message: string;
          nodeId?: string;
        }> = [];

        const { nodes, edges } = ctx.diagram;
        const getStereo = (name: string) => ctx.diagram.getStereotype(name);

        // ── 1. Empty graph ─────────────────────────────────────
        if (nodes.length === 0) {
          errors.push({
            code: "EMPTY_GRAPH",
            message:
              "The diagram is empty. Add at least one Input node and one Loss node.",
          });
          return {
            contents: [
              {
                uri: "nnmodelling://validation",
                mimeType: "application/json",
                text: JSON.stringify(
                  {
                    valid: false,
                    errors,
                    warnings,
                    timestamp: new Date().toISOString(),
                  },
                  null,
                  2,
                ),
              },
            ],
          };
        }

        // ── 2. Input node count ────────────────────────────────
        const inputNodeIds = new Set<string>();
        for (const n of nodes) {
          const stereoName = getStereoName(n);
          if (stereoName) {
            const stereo = getStereo(stereoName);
            if (stereo?.isInput) inputNodeIds.add(n.id);
          } else if (
            n.data &&
            (n.data as Record<string, unknown>).stereotype === "Input"
          ) {
            inputNodeIds.add(n.id);
          }
        }

        if (inputNodeIds.size === 0) {
          errors.push({
            code: "MISSING_INPUT",
            message:
              "No Input node found. Every diagram must have exactly one Input node.",
          });
        } else if (inputNodeIds.size > 1) {
          errors.push({
            code: "MULTIPLE_INPUTS",
            message: `Found ${inputNodeIds.size} Input nodes. Every diagram must have exactly one Input node.`,
          });
        }

        // ── 3. Loss node presence ──────────────────────────────
        const nodesWithOutgoing = new Set(edges.map((e) => e.source));
        const terminalNodes = nodes.filter(
          (n) => !nodesWithOutgoing.has(n.id),
        );

        let hasLossNode = false;
        for (const n of terminalNodes) {
          const stereoName = getStereoName(n);
          if (stereoName) {
            const stereo = getStereo(stereoName);
            if (stereo?.isLoss) {
              hasLossNode = true;
              break;
            }
          }
        }

        if (!hasLossNode) {
          warnings.push({
            code: "NO_LOSS_NODE",
            message:
              "No Loss node (terminal node with Loss category) detected. " +
              "The diagram will compile but may not produce training metrics. " +
              "Add a loss node (e.g. CrossEntropyLoss, MSELoss) as the final output.",
          });
        }

        // ── 4. Disconnected / orphan nodes ─────────────────────
        const nodesWithIncoming = new Set(edges.map((e) => e.target));

        for (const n of nodes) {
          const stereoName = getStereoName(n);
          if (stereoName) {
            const stereo = getStereo(stereoName);
            if (stereo?.isInput) continue;
            if (stereo?.isLoss) continue;
          }

          const hasIncoming = nodesWithIncoming.has(n.id);
          const hasOutgoing = nodesWithOutgoing.has(n.id);

          if (!hasIncoming && !hasOutgoing) {
            warnings.push({
              code: "ORPHAN_NODE",
              message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no incoming or outgoing connections.`,
              nodeId: n.id,
            });
          } else if (!hasIncoming) {
            warnings.push({
              code: "DISCONNECTED_INPUT",
              message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no incoming connections.`,
              nodeId: n.id,
            });
          } else if (!hasOutgoing) {
            const sName = getStereoName(n);
            if (!sName || !getStereo(sName)?.isLoss) {
              warnings.push({
                code: "DISCONNECTED_OUTPUT",
                message: `Node "${(n.data as Record<string, unknown>)?.name ?? n.id}" has no outgoing connections but is not a Loss node.`,
                nodeId: n.id,
              });
            }
          }
        }

        // ── 5. Cycle detection ─────────────────────────────────
        if (!isCycleFree(ctx.diagram)) {
          errors.push({
            code: "GRAPH_CYCLE",
            message:
              "The diagram contains one or more cycles. " +
              "Cycle detection uses Kahn's algorithm on the full edge set. " +
              "Cycles are not allowed in the computation graph.",
          });
        }

        return {
          contents: [
            {
              uri: "nnmodelling://validation",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  valid: errors.length === 0,
                  errors,
                  warnings,
                  timestamp: new Date().toISOString(),
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://statistics ─────────────────────────────────
    {
      uri: "nnmodelling://statistics",
      name: "Graph Statistics",
      description:
        "Aggregate statistics about the current graph: node/edge counts, " +
        "max depth via BFS, average fan-out, and cycle detection",
      mimeType: "application/json",
      async read() {
        const { nodes, edges } = ctx.diagram;
        const getStereo = (name: string) => ctx.diagram.getStereotype(name);

        let moduleCount = 0;
        let joinCount = 0;
        let subflowCount = 0;
        let inputCount = 0;
        let lossCount = 0;

        for (const node of nodes) {
          if (node.type === "join") {
            joinCount++;
          } else if (node.type === "subflow") {
            subflowCount++;
          } else {
            moduleCount++;
          }
          const stereoName = getStereoName(node);
          if (stereoName) {
            const stereo = getStereo(stereoName);
            if (stereo?.isInput) inputCount++;
            if (stereo?.isLoss) lossCount++;
          }
        }

        return {
          contents: [
            {
              uri: "nnmodelling://statistics",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  nodeCount: nodes.length,
                  edgeCount: edges.length,
                  moduleCount,
                  joinCount,
                  subflowCount,
                  inputCount,
                  lossCount,
                  maxDepth: computeMaxDepth(ctx.diagram, getStereo),
                  avgFanOut: computeAvgFanOut(ctx.diagram, getStereo),
                  cycleFree: isCycleFree(ctx.diagram),
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://conversion ────────────────────────────────
    {
      uri: "nnmodelling://conversion",
      name: "NNTree Compilation Output",
      description:
        "Latest NNTree compilation output. Compiles the current diagram " +
        "into an NNTree representation suitable for conversion and training. " +
        "Returns compilation results only; pipeline execution status is " +
        "in nnmodelling://conversion/status.",
      mimeType: "application/json",
      async read() {
        try {
          const nntree = new NNTree(ctx.diagram);
          const json = nntree.toJson();
          const parsed = JSON.parse(json);

          let subflowCount = 0;
          for (const [, node] of nntree.nodes) {
            if (node.isSubflow()) subflowCount++;
          }

          return {
            contents: [
              {
                uri: "nnmodelling://conversion",
                mimeType: "application/json",
                text: JSON.stringify(
                  {
                    success: true,
                    nntree: json,
                    root: nntree.root,
                    nodeCount: nntree.nodes.size,
                    subflowCount,
                    lossNodeType: nntree.lossNode?.stereotype ?? null,
                    timestamp: new Date().toISOString(),
                  },
                  null,
                  2,
                ),
              },
            ],
          };
        } catch (err: unknown) {
          const message =
            err instanceof Error ? err.message : "Unknown compilation error";
          return {
            contents: [
              {
                uri: "nnmodelling://conversion",
                mimeType: "application/json",
                text: JSON.stringify(
                  {
                    success: false,
                    error: message,
                    timestamp: new Date().toISOString(),
                  },
                  null,
                  2,
                ),
              },
            ],
          };
        }
      },
    },

    // ── nnmodelling://conversion/status ─────────────────────────
    {
      uri: "nnmodelling://conversion/status",
      name: "Conversion Pipeline Status",
      description:
        "Status of the conversion pipeline. Shows whether the diagram " +
        "can be compiled and converted. Pipeline execution results (outputDir, " +
        "configFiles) are populated after calling execute_conversion.",
      mimeType: "application/json",
      async read() {
        // Attempt compilation to report current viability
        let compilationOk = false;
        let compilationError: string | null = null;

        try {
          const nntree = new NNTree(ctx.diagram);
          nntree.toJson();
          compilationOk = true;
        } catch (err: unknown) {
          compilationError =
            err instanceof Error ? err.message : "Unknown compilation error";
        }

        return {
          contents: [
            {
              uri: "nnmodelling://conversion/status",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  ready: compilationOk,
                  pipelineExecuted: false,
                  compilationOk,
                  compilationError,
                  diagramNodeCount: ctx.diagram.nodes.length,
                  diagramEdgeCount: ctx.diagram.edges.length,
                  timestamp: new Date().toISOString(),
                  message: compilationOk
                    ? "Diagram is compilable. Use execute_conversion tool to run the full pipeline."
                    : `Diagram cannot be compiled: ${compilationError}. Fix validation errors first.`,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://history ────────────────────────────────────
    {
      uri: "nnmodelling://history",
      name: "Undo/Redo History Status",
      description: "Current undo/redo stack sizes and maximum depth configuration",
      mimeType: "application/json",
      async read() {
        const status = ctx.history.getStatus();

        return {
          contents: [
            {
              uri: "nnmodelling://history",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  undoCount: status.undoCount,
                  redoCount: status.redoCount,
                  maxUndoDepth: status.maxUndoDepth,
                  canUndo: status.undoCount > 0,
                  canRedo: status.redoCount > 0,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },

    // ── nnmodelling://events ─────────────────────────────────────
    {
      uri: "nnmodelling://events",
      name: "Event Log",
      description:
        "Accumulated domain events from the EventBus ring buffer. " +
        "Returns the last 1000 events with their sequence numbers and timestamps.",
      mimeType: "application/json",
      async read() {
        // Get all buffered events from the ring buffer
        const allEvents = ctx.diagram.events.getEventsSince(0);
        const latestSeq = ctx.diagram.events.getCurrentSeq();

        return {
          contents: [
            {
              uri: "nnmodelling://events",
              mimeType: "application/json",
              text: JSON.stringify(
                {
                  events: allEvents.map((e) => ({
                    seq: e.seq,
                    type: e.type,
                    timestamp: e.timestamp,
                    transactionId: e.transactionId ?? undefined,
                    payload: e.payload,
                  })),
                  count: allEvents.length,
                  latestSeq,
                  bufferSize: 1000,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    },
  ];
}
