/**
 * @file Type inference engine for NNModelling diagrams.
 *
 * DATA-DRIVEN constraint-based type inference.  No hardcoded module names,
 * no category checks.  Everything is driven by the `type_signature` field
 * in stereotype JSON files.
 *
 * Phase 1 limitations:
 * - Wildcard in pattern must be the last non-terminal element (no lookahead).
 * - Join type inference is a TODO (Phase 3).
 * - Subflow type inference is a TODO (Phase 4).
 */

import type { Node, Edge } from "@xyflow/svelte";
import { Diagram } from "../Diagram.svelte";
import type { Stereotype } from "../stereotype";
import type {
  TensorType,
  ShapeDimension,
  ShapeDimPattern,
  ShapePattern,
  TypeSignature,
  TypeError,
  NodeTypeAnnotation,
  TypeResult,
  TypeEnvironment,
  ParamResolution,
} from "./tensortypes";

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface PatternMatchResult {
  success: true;
  /** New or updated symbolic bindings (merged into env after match). */
  bindings: TypeEnvironment;
  /** Dimensions captured by wildcards, in order. */
  captured: ShapeDimension[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export class TypeEngine {
  /**
   * Full-graph type inference.
   *
   * 1. Build topological order of all top-level nodes (parentId === undefined).
   * 2. Traverse nodes in topological order, inferring types.
   * 3. Return all annotations, errors, and a summary `ok` flag.
   */
  static infer(diagram: Diagram): TypeResult {
    const annotations = new Map<string, NodeTypeAnnotation>();
    const errors: TypeError[] = [];
    const env: TypeEnvironment = new Map();

    // Only top-level nodes (subflow internals deferred to Phase 4)
    const topLevelNodes = diagram.nodes.filter((n) => !n.parentId);
    const sortedIds = this.topologicalSort(topLevelNodes, diagram.edges);

    // Cycle detection
    if (sortedIds.length < topLevelNodes.length) {
      errors.push({
        nodeId: "",
        message: "graph contains cycle, partial type inference",
        severity: "warning",
      });
    }

    // Build a fast id→node lookup
    const nodesById = new Map<string, Node>();
    for (const n of diagram.nodes) {
      nodesById.set(n.id, n);
    }

    for (const nodeId of sortedIds) {
      const node = nodesById.get(nodeId);
      if (!node) continue;

      // ── Step a: Get Stereotype ────────────────────────────────────
      const stereoName: string = (node.data as Record<string, unknown>)
        .stereotype as string;
      const stereotype = diagram.getStereotype(stereoName);

      // ── Step b: No type signature — with generic subflow handling ──
      if (!stereotype || !stereotype.typeSignature) {
        // Generic subflow: has children nodes but no stereotype with type_signature
        const hasInternalNodes = diagram.nodes.some((n) => n.parentId === nodeId);
        if (hasInternalNodes) {
          const localParams: Record<string, unknown> = (
            node.data as Record<string, unknown>
          ).params as Record<string, unknown>;
          // Compute input type inline (inputType not yet declared at this point)
          const incEdges = diagram.edges.filter((e) => e.target === nodeId);
          let subflowInputType: TensorType | undefined;
          if (incEdges.length === 1) {
            const srcAnn = annotations.get(incEdges[0].source);
            if (srcAnn) subflowInputType = srcAnn.outputType;
          } else if (incEdges.length > 1) {
            // Use first predecessor's type for subflow
            const srcAnn = annotations.get(incEdges[0].source);
            if (srcAnn) subflowInputType = srcAnn.outputType;
          }
          const result = this.inferSubflow(
            nodeId,
            subflowInputType,
            diagram,
            localParams ?? {},
            env,
            annotations,
            errors,
          );
          if (isTensorType(result)) {
            annotations.set(nodeId, {
              nodeId,
              outputType: result,
              ...(subflowInputType !== undefined ? { inputType: subflowInputType } : {}),
            });
            for (const dim of result.shape) {
              if (dim.kind === "symbolic" && !env.has(dim.name)) {
                env.set(dim.name, dim);
              }
            }
          } else {
            if (!result.nodeId) result.nodeId = nodeId;
            errors.push(result);
          }
          continue;
        }

        errors.push({
          nodeId,
          message: `No type signature for "${stereoName}"`,
          severity: "warning",
        });
        annotations.set(nodeId, {
          nodeId,
          outputType: { shape: [], dtype: "unknown" },
        });
        continue;
      }

      // ── Step c: Get resolved params ──────────────────────────────
      const params: Record<string, unknown> = (
        node.data as Record<string, unknown>
      ).params as Record<string, unknown>;

      const sig = stereotype.typeSignature;

      // ── Step d: Determine input type(s) ──────────────────────────
      const incomingEdges = diagram.edges.filter((e) => e.target === nodeId);

      let inputType: TensorType | undefined;
      let inputTypes: TensorType[] | undefined;

      if (stereotype.isInput) {
        // Source node — no incoming edges expected
        inputType = undefined;
      } else if (sig.kind === "join") {
        // Join nodes have multiple inputs — collect them
        const collected: TensorType[] = [];
        let allFound = true;
        // Sort by targetHandle for deterministic ordering (in-0, in-1, …)
        const sortedEdges = [...incomingEdges].sort((a, b) => {
          const ha = a.targetHandle ?? "in-0";
          const hb = b.targetHandle ?? "in-0";
          return ha.localeCompare(hb);
        });
        for (const e of sortedEdges) {
          const srcAnn = annotations.get(e.source);
          if (srcAnn) {
            collected.push(srcAnn.outputType);
          } else {
            allFound = false;
          }
        }
        if (!allFound) {
          errors.push({
            nodeId,
            message: "Some predecessors have no type annotation",
            severity: "warning",
          });
          continue;
        }
        inputTypes = collected.length > 0 ? collected : undefined;
        // Pass inputTypes to inferNode for join processing
        inputType = undefined;
      } else if (incomingEdges.length === 0) {
        // No predecessor and not Input — floating node, skip silently
        continue;
      } else if (incomingEdges.length === 1) {
        const srcAnn = annotations.get(incomingEdges[0].source);
        if (!srcAnn) {
          errors.push({
            nodeId,
            message: `Predecessor "${incomingEdges[0].source}" has no type annotation`,
            severity: "warning",
          });
          continue;
        }
        inputType = srcAnn.outputType;
      } else {
        // Multiple predecessors but not a join — gather what we can
        const collected: TensorType[] = [];
        let allFound = true;
        for (const e of incomingEdges) {
          const srcAnn = annotations.get(e.source);
          if (srcAnn) {
            collected.push(srcAnn.outputType);
          } else {
            allFound = false;
          }
        }
        if (!allFound) {
          errors.push({
            nodeId,
            message: "Some predecessors have no type annotation",
            severity: "warning",
          });
          continue;
        }
        // Use the first predecessor's type for single-input pattern matching
        inputType = collected[0];
        inputTypes = collected;
      }

      // ── Step e: Call inferNode ───────────────────────────────────
      const result = this.inferNode(
        inputType,
        stereotype,
        params,
        env,
        inputTypes,
        diagram,
        nodeId,
        annotations,
        errors,
      );

      // ── Steps f/g: Handle result ─────────────────────────────────
      if (isTensorType(result)) {
        const annotation: NodeTypeAnnotation = {
          nodeId,
          outputType: result,
        };
        if (inputType !== undefined) {
          annotation.inputType = inputType;
        }
        if (inputTypes !== undefined && inputTypes.length > 1) {
          annotation.inputTypes = inputTypes;
        }
        annotations.set(nodeId, annotation);

        // Update environment with any new symbolic bindings from the output
        for (const dim of result.shape) {
          if (dim.kind === "symbolic" && !env.has(dim.name)) {
            env.set(dim.name, dim);
          }
        }
      } else {
        // Patch the nodeId if the callee left it empty
        if (!result.nodeId) {
          result.nodeId = nodeId;
        }
        errors.push(result);
      }
    }

    return {
      ok: errors.every((e) => e.severity !== "error"),
      annotations,
      errors,
    };
  }

  /**
   * Single-node type inference.
   *
   * Takes an input type, stereotype (with its type_signature), node params,
   * and the current environment, and returns either a concrete output type
   * or a TypeError.
   */
  static inferNode(
    inputType: TensorType | undefined,
    stereotype: Stereotype,
    params: Record<string, unknown>,
    env: TypeEnvironment,
    inputTypes?: TensorType[],
    diagram?: Diagram,
    nodeId?: string,
    annotations?: Map<string, NodeTypeAnnotation>,
    errors?: TypeError[],
  ): TensorType | TypeError {
    // Safety check: type signature must exist
    const sig = stereotype.typeSignature;
    if (!sig) {
      return {
        nodeId: "",
        message: `No type signature for "${stereotype.name}"`,
        severity: "warning",
      } satisfies TypeError;
    }

    switch (sig.kind) {
      // ── Module kind ────────────────────────────────────────────
      case "module": {
        const inputPattern = sig.input as ShapePattern;

        // Source node with empty pattern → skip input matching
        if (inputType === undefined) {
          if (inputPattern.length === 0) {
            // Produce output from pattern alone (e.g. Input node)
            const outputDims = this.resolvePattern(
              sig.output,
              params,
              env,
              [],
            );
            const dtype = sig.dtype?.output ?? "unknown";
            return { shape: outputDims, dtype } satisfies TensorType;
          }

          // Non-empty pattern but no input → error
          return {
            nodeId: "",
            message: `"${stereotype.name}" expects input but has no predecessor`,
            severity: "error",
          } satisfies TypeError;
        }

        // Normal case: match input against pattern
        const matchResult = this.patternMatch(
          inputType.shape,
          inputPattern,
          params,
          env,
        );

        if (isTypeError(matchResult)) {
          return matchResult;
        }

        // Merge new bindings into env for downstream use
        for (const [key, value] of matchResult.bindings) {
          env.set(key, value);
        }

        const outputDims = this.resolvePattern(
          sig.output,
          params,
          env,
          matchResult.captured,
        );
        const dtype = sig.dtype?.output ?? inputType.dtype;

        return { shape: outputDims, dtype } satisfies TensorType;
      }

      // ── Join kind — Phase 3 ──────────────────────────────────
      case "join": {
        const inputPatterns = sig.input as ShapePattern[];

        if (!inputTypes || inputTypes.length === 0) {
          return {
            nodeId: "",
            message: `Join node "${stereotype.name}" has no inputs`,
            severity: "error",
          } satisfies TypeError;
        }

        if (inputPatterns.length !== inputTypes.length) {
          return {
            nodeId: "",
            message: `Join node "${stereotype.name}" expects ${inputPatterns.length} inputs but got ${inputTypes.length}`,
            severity: "error",
          } satisfies TypeError;
        }

        // Step 1: Match each input pattern against corresponding input type
        const allBindings: TypeEnvironment[] = [];
        const allCaptured: ShapeDimension[][] = [];

        for (let k = 0; k < inputPatterns.length; k++) {
          const pat = inputPatterns[k];
          const inp = inputTypes[k];
          if (!inp) continue;

          const matchResult = this.patternMatch(inp.shape, pat, params, env);
          if (isTypeError(matchResult)) {
            return {
              nodeId: "",
              message: `Input ${k} mismatch: ${matchResult.message}`,
              severity: matchResult.severity,
            } satisfies TypeError;
          }

          allCaptured.push(matchResult.captured);
          allBindings.push(matchResult.bindings);
        }

        // ── Verify all inputs have compatible shapes (captured dims match) ──
        // For joins with wildcard patterns (e.g. Addition, element-wise ops),
        // all inputs must have identical shapes.  We verify by comparing captured
        // dims across inputs: same length and equal dimension values.
        // This check is SKIPPED for concat joins (where the concat dim differs).
        if (allCaptured.length >= 2 && !sig.constraints?.concat) {
          const first = allCaptured[0];
          for (let k = 1; k < allCaptured.length; k++) {
            const other = allCaptured[k];
            if (first.length !== other.length) {
              return {
                nodeId: "",
                message: `Input ${k} shape length mismatch: expected ${first.length} dims, got ${other.length}`,
                severity: "error",
              } satisfies TypeError;
            }
            for (let d = 0; d < first.length; d++) {
              if (!dimEqual(first[d], other[d])) {
                return {
                  nodeId: "",
                  message: `Input ${k} dimension ${d} mismatch: ${this.describeDim(first[d])} vs ${this.describeDim(other[d])}`,
                  severity: "error",
                } satisfies TypeError;
              }
            }
          }
        }

        // Step 2: Merge symbolic bindings across all inputs (unification)
        const mergedEnv: TypeEnvironment = new Map(env);
        for (const bindings of allBindings) {
          for (const [name, dim] of bindings) {
            const existing = mergedEnv.get(name);
            if (existing !== undefined) {
              if (!dimEqual(existing, dim)) {
                return {
                  nodeId: "",
                  message: `Symbolic $${name} bound to conflicting values: ${this.describeDim(existing)} vs ${this.describeDim(dim)}`,
                  severity: "error",
                } satisfies TypeError;
              }
            } else {
              mergedEnv.set(name, dim);
            }
          }
        }

        // Merge mergedEnv into env for downstream use
        for (const [key, value] of mergedEnv) {
          env.set(key, value);
        }

        // Step 3: Compute output shape
        let outputDims: ShapeDimension[];

        if (sig.constraints?.concat) {
          // Concat join: sum dims on the concat axis
          const concatDim = this.resolveConcatDim(
            sig.constraints.concat.dim,
            params,
            inputTypes,
          );
          if (concatDim === undefined) {
            return {
              nodeId: "",
              message: `Invalid concat dim "${sig.constraints.concat.dim}" for "${stereotype.name}"`,
              severity: "error",
            } satisfies TypeError;
          }
          outputDims = this.resolveConcatOutput(inputTypes, concatDim);

          // Verify non-concat dims match across all inputs
          for (let idx = 1; idx < inputTypes.length; idx++) {
            for (let d = 0; d < inputTypes[0].shape.length; d++) {
              if (d === concatDim) continue;
              if (!dimEqual(inputTypes[0].shape[d], inputTypes[idx].shape[d])) {
                return {
                  nodeId: "",
                  message: `Concat input ${idx} dimension ${d} mismatch: expected ${this.describeDim(inputTypes[0].shape[d])}, got ${this.describeDim(inputTypes[idx].shape[d])}`,
                  severity: "error",
                } satisfies TypeError;
              }
            }
          }
        } else {
          // Standard join: resolve output pattern with merged env
          // Use only the first input's captured dims for output resolution;
          // other inputs are for verification only.
          const outputCaptured = allCaptured.length > 0 ? allCaptured[0] : [];
          outputDims = this.resolvePattern(
            sig.output,
            params,
            mergedEnv,
            outputCaptured,
          );
        }

        const dtype = sig.dtype?.output ?? inputTypes[0].dtype;

        return { shape: outputDims, dtype } satisfies TensorType;
      }

      // ── Subflow kind (Phase 4) ─────────────────────────────────
      case "subflow": {
        const stereoName = stereotype.name;
        // Shared maps for writing internal node annotations/errors
        const subflowAnnotations = annotations ?? new Map<string, NodeTypeAnnotation>();
        const subflowErrors = errors ?? [];

        // ── Repeat: shape-preserving ──────────────────────────
        if (stereoName === "Repeat") {
          if (!inputType) {
            return {
              nodeId: "",
              message: "Repeat subflow has no input",
              severity: "error",
            } satisfies TypeError;
          }
          // Repeat executes N times → output shape equals input shape
          return {
            shape: inputType.shape.map((d) => ({ ...d })),
            dtype: inputType.dtype,
          } satisfies TensorType;
        }

        // ── HorizontalRepeat: run internal graph then concat on last dim ──
        if (stereoName === "HorizontalRepeat") {
          if (!inputType) {
            return {
              nodeId: "",
              message: "HorizontalRepeat subflow has no input",
              severity: "error",
            } satisfies TypeError;
          }
          const nResolved = this.resolveParamRef("n", params);
          if (nResolved.status !== "resolved") {
            return {
              nodeId: "",
              message:
                "HorizontalRepeat requires parameter 'n' to be set",
              severity: "error",
            } satisfies TypeError;
          }
          // First run the internal subflow graph to get its output type
          if (!diagram || !nodeId) {
            return {
              nodeId: "",
              message:
                "HorizontalRepeat requires diagram context for recursive inference",
              severity: "error",
            } satisfies TypeError;
          }
          const internalResult = this.inferSubflow(
            nodeId,
            inputType,
            diagram,
            params,
            env,
            subflowAnnotations,
            subflowErrors,
          );
          if (!isTensorType(internalResult)) {
            return internalResult; // propagate subflow error (structural)
          }
          const n = nResolved.value;
          const newShape = internalResult.shape.map((d, i) => {
            if (
              i === internalResult.shape.length - 1 &&
              d.kind === "const"
            ) {
              return { kind: "const" as const, value: d.value * n };
            }
            return { ...d };
          });
          return {
            shape: newShape,
            dtype: internalResult.dtype,
          } satisfies TensorType;
        }

        // ── Generic subflow: recursive inference ────────────
        if (!diagram || !nodeId) {
          return {
            nodeId: "",
            message: `Subflow "${stereoName}" requires diagram context for recursive inference`,
            severity: "error",
          } satisfies TypeError;
        }
        return this.inferSubflow(
          nodeId,
          inputType,
          diagram,
          params,
          env,
          subflowAnnotations,
          subflowErrors,
        );
      }

      default:
        return {
          nodeId: "",
          message: `Unknown type signature kind "${(sig as TypeSignature).kind}" for "${stereotype.name}"`,
          severity: "warning",
        } satisfies TypeError;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Subflow inference (Phase 4)
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Recursive type inference for a generic subflow container.
   *
   * 1. Collects internal child nodes (parentId === subflowNodeId).
   * 2. Finds the internal Input node and injects the external input type.
   * 3. Runs topological sort on internal nodes.
   * 4. Calls inferNode for each internal node (recursing into nested subflows).
   * 5. Returns the output type of the first exit node.
   */
  private static inferSubflow(
    subflowNodeId: string,
    externalInputType: TensorType | undefined,
    diagram: Diagram,
    params: Record<string, unknown>,
    env: TypeEnvironment,
    annotations: Map<string, NodeTypeAnnotation>,
    errors: TypeError[],
  ): TensorType | TypeError {
    // 1. Collect internal nodes
    const internalNodes = diagram.nodes.filter(
      (n) => n.parentId === subflowNodeId,
    );
    if (internalNodes.length === 0) {
      return {
        nodeId: subflowNodeId,
        message: "Subflow has no internal nodes",
        severity: "warning",
      } satisfies TypeError;
    }

    // 2. Collect internal edges (both ends inside the subflow)
    const internalNodeIds = new Set(internalNodes.map((n) => n.id));
    const internalEdges = diagram.edges.filter(
      (e) =>
        internalNodeIds.has(e.source) && internalNodeIds.has(e.target),
    );

    // 3. Find entry node (internal Input or topological source)
    let entryNode: Node | undefined = internalNodes.find((n) => {
      const stereo = diagram.getStereotype(
        (n.data as Record<string, unknown>).stereotype as string,
      );
      return stereo?.isInput;
    });

    // Fallback: if no Input node, use node(s) with no internal incoming edges
    if (!entryNode) {
      const sources = internalNodes.filter(
        (n) => !internalEdges.some((e) => e.target === n.id),
      );
      if (sources.length >= 1) {
        entryNode = sources[0];
      }
    }

    if (!entryNode) {
      return {
        nodeId: subflowNodeId,
        message: "Subflow has no internal Input node",
        severity: "error",
      } satisfies TypeError;
    }

    // 4. Find exit node(s) — internal nodes with no internal outgoing edges.
    //    The entry node CAN be an exit too (single-node subflows).
    let exitNodes = internalNodes.filter(
      (n) => !internalEdges.some((e) => e.source === n.id),
    );
    if (exitNodes.length === 0) {
      return {
        nodeId: subflowNodeId,
        message: "Could not determine subflow exit node",
        severity: "error",
      } satisfies TypeError;
    }

    // 5. Seed annotations for Input entry node.
    //    For Input entry nodes: seed with externalInputType as output
    //    (Input just passes through). For other entry nodes (e.g. nested subflow
    //    containers), we don't seed — the external input is passed as input to
    //    the entry node during processing below.
    const entryStereo = diagram.getStereotype(
      (entryNode.data as Record<string, unknown>).stereotype as string,
    );
    const isEntryInput = entryStereo?.isInput ?? false;
    if (externalInputType && isEntryInput) {
      annotations.set(entryNode.id, {
        nodeId: entryNode.id,
        outputType: externalInputType,
      });
    }

    // 6. Topological sort internal nodes
    const sortedIds = this.topologicalSort(internalNodes, internalEdges);

    // 7. Build fast id→node lookup
    const nodesById = new Map<string, Node>();
    for (const n of internalNodes) {
      nodesById.set(n.id, n);
    }

    const localEnv: TypeEnvironment = new Map(env);

    // 8. Walk internal nodes (skip Input entry node if already seeded)
    for (const internalNodeId of sortedIds) {
      const isEntry = internalNodeId === entryNode.id;
      if (isEntry && isEntryInput && externalInputType) continue; // Already seeded

      const n = nodesById.get(internalNodeId);
      if (!n) continue;

      // 8a. Get stereotype
      const stereoName = (n.data as Record<string, unknown>)
        .stereotype as string;
      const stereotype = diagram.getStereotype(stereoName);

      // 8b. No type signature — check for nested subflow or skip
      if (!stereotype || !stereotype.typeSignature) {
        const hasInternalChildren = diagram.nodes.some(
          (cn) => cn.parentId === internalNodeId,
        );
        if (hasInternalChildren) {
          // Recursive subflow inference for nested generic subflow
          // Compute input type: entry node gets externalInputType,
          // other nodes get it from internal predecessor annotations
          let nestedInputType: TensorType | undefined;
          if (isEntry && externalInputType) {
            nestedInputType = externalInputType;
          } else {
            const incomingEdges = internalEdges.filter(
              (e) => e.target === internalNodeId,
            );
            if (incomingEdges.length === 1) {
              const srcAnn = annotations.get(
                incomingEdges[0].source,
              );
              if (srcAnn)
                nestedInputType = srcAnn.outputType;
            }
          }
          const nestedResult = this.inferSubflow(
            internalNodeId,
            nestedInputType,
            diagram,
            (n.data as Record<string, unknown>)
              .params as Record<string, unknown> ?? {},
            localEnv,
            annotations,
            errors,
          );
          if (isTensorType(nestedResult)) {
            annotations.set(internalNodeId, {
              nodeId: internalNodeId,
              outputType: nestedResult,
            });
          } else {
            // Structural error from nested subflow (no Input, no exit, etc.)
            // Return as before — shows on the subflow container
            return {
              nodeId: subflowNodeId,
              message: `[Subflow] ${nestedResult.message}`,
              severity: nestedResult.severity,
            } satisfies TypeError;
          }
        } else {
          // Non-subflow node without type signature — warning
          annotations.set(internalNodeId, {
            nodeId: internalNodeId,
            outputType: { shape: [], dtype: "unknown" },
          });
        }
        continue;
      }

      // 8c. Get resolved params
      const localParams: Record<string, unknown> = (
        n.data as Record<string, unknown>
      ).params as Record<string, unknown>;

      const sig = stereotype.typeSignature;

      // 8d. Determine input type from internal edges
      const localIncomingEdges = internalEdges.filter(
        (e) => e.target === internalNodeId,
      );

      let localInputType: TensorType | undefined;
      let localInputTypes: TensorType[] | undefined;

      // Entry node that is not an Input receives the external input type
      if (isEntry && !isEntryInput && externalInputType) {
        localInputType = externalInputType;
      } else if (stereotype.isInput) {
        localInputType = undefined;
      } else if (sig.kind === "join") {
        const collected: TensorType[] = [];
        const sortedEdges = [...localIncomingEdges].sort((a, b) => {
          const ha = a.targetHandle ?? "in-0";
          const hb = b.targetHandle ?? "in-0";
          return ha.localeCompare(hb);
        });
        for (const e of sortedEdges) {
          const srcAnn = annotations.get(e.source);
          if (srcAnn) collected.push(srcAnn.outputType);
        }
        localInputTypes =
          collected.length > 0 ? collected : undefined;
      } else if (localIncomingEdges.length === 1) {
        const srcAnn = annotations.get(
          localIncomingEdges[0].source,
        );
        if (srcAnn) localInputType = srcAnn.outputType;
      }

      // 8e. Call inferNode on the internal node (handles Repeat, HorizontalRepeat, module, join)
      const result = this.inferNode(
        localInputType,
        stereotype,
        localParams,
        localEnv,
        localInputTypes,
        diagram,
        internalNodeId,
        annotations,
        errors,
      );

      if (isTensorType(result)) {
        annotations.set(internalNodeId, {
          nodeId: internalNodeId,
          outputType: result,
        });
        for (const dim of result.shape) {
          if (dim.kind === "symbolic" && !localEnv.has(dim.name)) {
            localEnv.set(dim.name, dim);
          }
        }
      } else {
        // Push internal error with correct nodeId (the inferNode TypeError already
        // has nodeId="" — patch it to the internal node ID)
        if (!result.nodeId) {
          result.nodeId = internalNodeId;
        }
        errors.push(result);
        // Set fallback annotation so downstream internal nodes can continue
        annotations.set(internalNodeId, {
          nodeId: internalNodeId,
          outputType: { shape: [], dtype: "unknown" },
        });
      }
    }

    // 9. Return first exit node's output type
    const firstExit = exitNodes[0];
    const exitAnn = annotations.get(firstExit.id);
    if (!exitAnn) {
      return {
        nodeId: subflowNodeId,
        message: "Subflow exit node has no type annotation",
        severity: "error",
      } satisfies TypeError;
    }

    return {
      shape: [...exitAnn.outputType.shape],
      dtype: exitAnn.outputType.dtype,
    } satisfies TensorType;
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Formula resolution (Phase 2 — computed dimensions)
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Resolve a dimension formula with concrete numeric arguments.
   *
   * Supported formulas:
   * - `conv2d_hw`:  H_out = floor((H + 2*pad - dilation*(kernel-1) - 1) / stride + 1)
   * - `pool2d_hw`:  H_out = floor((H + 2*pad - kernel) / stride + 1)
   * - `flatten_prod`: product of all args (dimensions being flattened)
   *
   * Returns `undefined` for unknown formulas.
   */
  private static resolveFormula(
    formula: string,
    args: number[],
  ): number | undefined {
    switch (formula) {
      case "conv2d_hw": {
        const [h, k, s, p, d] = args;
        return Math.floor((h + 2 * p - d * (k - 1) - 1) / s + 1);
      }
      case "pool2d_hw": {
        const [h, k, s, p] = args;
        return Math.floor((h + 2 * p - k) / s + 1);
      }
      case "flatten_prod": {
        return args.reduce((a, b) => a * b, 1);
      }
      case "upsample_hw": {
        const [h, scale] = args;
        return h * scale;
      }
      default:
        return undefined;
    }
  }

  /**
   * Resolve the concatenation dimension from a constraint like "params.dim".
   * Handles negative dims by wrapping from the end of the first input shape.
   */
  private static resolveConcatDim(
    dimSpec: string,
    params: Record<string, unknown>,
    inputTypes: TensorType[],
  ): number | undefined {
    const parts = dimSpec.split(".");
    if (parts.length === 2 && parts[0] === "params") {
      const dim = this.resolveParamRef(parts[1], params);
      if (dim.status !== 'resolved') return undefined;
      if (dim.value < 0 && inputTypes.length > 0) {
        // Wrap negative dim (e.g. -1 → last dim)
        return inputTypes[0].shape.length + dim.value;
      }
      return dim.value;
    }
    // Direct number
    const parsed = Number(dimSpec);
    if (!isNaN(parsed)) return parsed;
    return undefined;
  }

  /**
   * Compute the output shape for a Concat join.
   * Uses the first input's shape as a template and sums values on concatDim.
   * If any input's concat dim is non-const, falls back to the template.
   */
  private static resolveConcatOutput(
    inputTypes: TensorType[],
    concatDim: number,
  ): ShapeDimension[] {
    if (inputTypes.length === 0) return [];
    const template = inputTypes[0];
    if (concatDim < 0 || concatDim >= template.shape.length) {
      return template.shape.map((d) => ({ ...d }));
    }

    // Sum const values on concat dim across all inputs
    let total = 0;
    let allConst = true;
    for (const inp of inputTypes) {
      const dim = inp.shape[concatDim];
      if (dim.kind === "const") {
        total += dim.value;
      } else {
        allConst = false;
        break;
      }
    }

    const shape = template.shape.map((d) => ({ ...d }));
    if (allConst) {
      shape[concatDim] = { kind: "const" as const, value: total };
    }
    return shape;
  }

  /**
   * Resolve a single computed-arg string to a concrete number.
   *
   * - If the arg starts with `$`, it is a symbolic reference resolved from env.
   * - If the arg is `$*`, it expands to the product of all captured dims.
   * - Otherwise it is a parameter reference resolved from `params`.
   *
   * @param arg - The argument string to resolve (e.g. "$B", "$*", "kernel_size").
   * @param env - Symbolic environment with bound dimension values.
   * @param params - Node parameter map.
   * @param captured - Wildcard-captured dimensions from pattern matching.
   * @returns `{ resolved: true; value: number }` on success, `{ resolved: false }` otherwise.
   */
  private static resolveComputedArg(
    arg: string,
    env: TypeEnvironment,
    params: Record<string, unknown>,
    captured: ShapeDimension[],
  ): { resolved: true; value: number } | { resolved: false } {
    // ── Special wildcard reference: * ($ stripped during parsing) ──
    if (arg === "*") {
      let product = 1;
      for (const dim of captured) {
        if (dim.kind === "const") {
          product *= dim.value;
        } else {
          // One captured dim is non-const -> can't compute product
          return { resolved: false };
        }
      }
      return { resolved: true, value: product };
    }

    // ── Symbolic reference (e.g. B, H, W — $ stripped during parsing)
    const dim = env.get(arg);
    if (dim && dim.kind === "const") {
      return { resolved: true, value: dim.value };
    }

    // ── Parameter reference (e.g. kernel_size, stride) ──────────────
    const val = this.resolveParamRef(arg, params);
    if (val.status === 'resolved') {
      return { resolved: true, value: val.value };
    }
    return { resolved: false };
  }

  // ───────────────────────────────────────────────────────────────────────────
  // Private helpers
  // ───────────────────────────────────────────────────────────────────────────

  /**
   * Kahn's topological sort.
   * Only processes edges where both source and target are in the given node list.
   * Returns node IDs in topological order.
   */
  private static topologicalSort(
    nodes: Node[],
    edges: Edge[],
  ): string[] {
    const nodeIds = new Set(nodes.map((n) => n.id));

    // Build adjacency list and in-degree map
    const adj = new Map<string, string[]>();
    const inDegree = new Map<string, number>();

    for (const n of nodes) {
      adj.set(n.id, []);
      inDegree.set(n.id, 0);
    }

    for (const e of edges) {
      if (nodeIds.has(e.source) && nodeIds.has(e.target)) {
        adj.get(e.source)!.push(e.target);
        inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
      }
    }

    // Queue of nodes with in-degree 0
    const queue: string[] = [];
    for (const [id, deg] of inDegree) {
      if (deg === 0) queue.push(id);
    }

    const sorted: string[] = [];
    while (queue.length > 0) {
      const id = queue.shift()!;
      sorted.push(id);

      for (const child of adj.get(id) ?? []) {
        const newDeg = (inDegree.get(child) ?? 1) - 1;
        inDegree.set(child, newDeg);
        if (newDeg === 0) queue.push(child);
      }
    }

    return sorted;
  }

  /**
   * Pattern matching — core algorithm.
   *
   * Matches a resolved input shape against a declared pattern, returning
   * symbolic bindings and captured wildcard dimensions.
   */
  private static patternMatch(
    inputDims: ShapeDimension[],
    pattern: ShapePattern,
    params: Record<string, unknown>,
    env: TypeEnvironment,
  ): PatternMatchResult | TypeError {
    // Count wildcards — Phase 1 allows at most one
    const wildcardCount = pattern.filter((p) => p.kind === "wildcard").length;
    if (wildcardCount > 1) {
      return {
        nodeId: "",
        message: "only one wildcard allowed per pattern in Phase 1",
        severity: "error",
      } satisfies TypeError;
    }

    let i = 0; // index into inputDims
    let j = 0; // index into pattern
    const bindings = new Map(env); // copy existing environment
    const captured: ShapeDimension[] = [];

    while (j < pattern.length) {
      const p = pattern[j];

      switch (p.kind) {
        // ── const ─────────────────────────────────────────────
        case "const": {
          if (i >= inputDims.length) {
            return {
              nodeId: "",
              message: `expected dim at position ${j}, got end of shape`,
              severity: "error",
            } satisfies TypeError;
          }
          const constDim = inputDims[i];
          if (
            constDim.kind !== "const" ||
            constDim.value !== p.value
          ) {
            return {
              nodeId: "",
              message: `dimension mismatch at position ${j}: expected ${p.value}, got ${this.describeDim(constDim)}`,
              severity: "error",
            } satisfies TypeError;
          }
          i++;
          j++;
          break;
        }

        // ── symbolic ─────────────────────────────────────────
        case "symbolic": {
          if (i >= inputDims.length) {
            return {
              nodeId: "",
              message: `expected dim at position ${j}, got end of shape`,
              severity: "error",
            } satisfies TypeError;
          }
          const existing = bindings.get(p.name);
          if (existing !== undefined) {
            // Already bound — unify
            if (!dimEqual(existing, inputDims[i])) {
              return {
                nodeId: "",
                message: `symbolic $${p.name} already bound to ${this.describeDim(existing)}, cannot unify with ${this.describeDim(inputDims[i])}`,
                severity: "error",
              } satisfies TypeError;
            }
          } else {
            // First occurrence — bind
            bindings.set(p.name, inputDims[i]);
          }
          i++;
          j++;
          break;
        }

        // ── param_ref ────────────────────────────────────────
        case "param_ref": {
          const resolved = this.resolveParamRef(p.name, params);
          if (resolved.status === 'unset') {
            // Param is "Undefined"/missing — treat as symbolic
            captured.push({ kind: "symbolic", name: `?${p.name}` });
            i++;
            j++;
          } else if (resolved.status === 'invalid') {
            // Param has a non-numeric value — type error
            return {
              nodeId: "",
              message: `parameter ${p.name} has invalid value "${resolved.value}", expected a number`,
              severity: "error",
            } satisfies TypeError;
          } else {
            // resolved.status === 'resolved'
            if (i >= inputDims.length) {
              return {
                nodeId: "",
                message: `expected dim at position ${j}, got end of shape`,
                severity: "error",
              } satisfies TypeError;
            }
            const paramDim = inputDims[i];
            if (
              paramDim.kind !== "const" ||
              paramDim.value !== resolved.value
            ) {
              return {
                nodeId: "",
                message: `dimension mismatch: param ${p.name}=${resolved.value}, got ${this.describeDim(paramDim)}`,
                severity: "error",
              } satisfies TypeError;
            }
            i++;
            j++;
          }
          break;
        }

        // ── computed (pass-through — no input validation) ────
        case "computed": {
          // Computed dims in the input pattern just require the
          // dimension to exist; the exact value is not validated
          // since computed dims are primarily output-side constraints.
          if (i >= inputDims.length) {
            return {
              nodeId: "",
              message: `expected dim at position ${j}, got end of shape`,
              severity: "error",
            } satisfies TypeError;
          }
          i++;
          j++;
          break;
        }

        // ── wildcard ─────────────────────────────────────────
        case "wildcard": {
          // Count how many non-wildcard pattern elements follow.
          // The wildcard consumes all input dims except those reserved
          // for subsequent required pattern elements.
          let remainingRequired = 0;
          for (let k = j + 1; k < pattern.length; k++) {
            if (pattern[k].kind !== "wildcard") remainingRequired++;
          }

          // How many input dims remain?
          const available = inputDims.length - i;
          const toConsume = Math.max(0, available - remainingRequired);

          // Consume the wildcard dims
          for (let c = 0; c < toConsume; c++) {
            captured.push(inputDims[i]);
            i++;
          }
          j++;
          break;
        }

        default:
          return {
            nodeId: "",
            message: `unknown pattern kind: ${(p as ShapeDimPattern).kind}`,
            severity: "error",
          } satisfies TypeError;
      }
    }

    // After pattern consumed, check no extra input dims remain
    if (i < inputDims.length) {
      return {
        nodeId: "",
        message: `expected ${pattern.length} dimensions in pattern, but input has ${inputDims.length} dims (extra dims at positions ${i}..${inputDims.length - 1})`,
        severity: "error",
      } satisfies TypeError;
    }

    return { success: true, bindings, captured };
  }

  /**
   * Resolve an output pattern to a concrete shape, substituting bindings
   * and captured wildcard dimensions.
   */
  private static resolvePattern(
    pattern: ShapePattern,
    params: Record<string, unknown>,
    env: TypeEnvironment,
    captured: ShapeDimension[],
  ): ShapeDimension[] {
    // Empty output pattern → terminal node (e.g. Loss nodes)
    if (pattern.length === 0) return [];

    const result: ShapeDimension[] = [];
    let capIdx = 0;

    for (const p of pattern) {
      switch (p.kind) {
        case "const":
          result.push({ kind: "const", value: p.value });
          break;

        case "symbolic": {
          const bound = env.get(p.name);
          if (bound !== undefined) {
            result.push(bound);
          } else {
            // Symbolic unbound — keep as symbolic (propagate)
            result.push({ kind: "symbolic", name: p.name });
          }
          break;
        }

        case "param_ref": {
          const val = this.resolveParamRef(p.name, params);
          if (val.status === 'resolved') {
            result.push({ kind: "const", value: val.value });
          } else {
            // unset or invalid — keep as symbolic (error already reported in patternMatch)
            result.push({ kind: "symbolic", name: `?${p.name}` });
          }
          break;
        }

        case "wildcard": {
          // Substitute all captured dimensions at this position
          while (capIdx < captured.length) {
            result.push(captured[capIdx]);
            capIdx++;
          }
          break;
        }

        case "computed": {
          // Try to resolve all formula arguments
          const argValues: number[] = [];
          let allResolved = true;
          for (const arg of p.args) {
            const resolved = this.resolveComputedArg(
              arg,
              env,
              params,
              captured,
            );
            if (resolved.resolved) {
              argValues.push(resolved.value);
            } else {
              allResolved = false;
              break;
            }
          }

          if (allResolved) {
            const value = this.resolveFormula(p.formula, argValues);
            if (value !== undefined) {
              result.push({ kind: "const", value });
            } else {
              // Unknown formula — keep as deferred computed
              result.push({
                kind: "computed",
                formula: p.formula,
                args: p.args,
              });
            }
          } else {
            // Can't resolve all args — keep as deferred computed
            result.push({
              kind: "computed",
              formula: p.formula,
              args: p.args,
            });
          }
          break;
        }

        default:
          // Unknown pattern kind — keep as unknown symbolic
          result.push({
            kind: "symbolic",
            name: `?unknown_${(p as ShapeDimPattern).kind}`,
          });
          break;
      }
    }

    // Push any unconsumed captured dims only if the output pattern has a wildcard
    // to consume them (e.g. ReLU [*]).  For patterns without wildcards (e.g.
    // SequencePool [$B, $D]), captured dims are intentionally dropped.
    const hasWildcard = pattern.some(p => p.kind === "wildcard");
    if (hasWildcard) {
      while (capIdx < captured.length) {
        result.push(captured[capIdx]);
        capIdx++;
      }
    }

    return result;
  }

  /**
   * Resolve a parameter reference from a node's parameter map.
   *
   * Handles both flat values (`params[name] === "784"`) and wrapped objects
   * (`params[name] === { value: "784", position: "top" }`).
   *
   * Returns a discriminated union:
   * - `{ status: 'unset' }` — parameter is "None" or truly missing
   * - `{ status: 'invalid', value }` — parameter has a non-numeric value ("Undefined", "", "cazz", etc.)
   * - `{ status: 'resolved', value }` — parameter resolved to a number
   */
  static resolveParamRef(
    name: string,
    params: Record<string, unknown>,
  ): ParamResolution {
    const raw = params[name];
    if (raw === undefined || raw === null) return { status: 'unset' };

    // Handle wrapped parameter objects: { value: "784", position: "top" }
    const val: unknown =
      typeof raw === "object" && raw !== null && "value" in raw
        ? (raw as Record<string, unknown>).value
        : raw;

    if (typeof val === "number") return { status: 'resolved', value: val };
    if (typeof val === "string") {
      if (val === "None") {
        return { status: 'unset' };
      }
      // "Undefined", "" (empty), and any other non-numeric string are invalid
      if (val === "Undefined" || val === "") {
        return { status: 'invalid', value: val };
      }
      const parsed = Number(val);
      if (isNaN(parsed)) {
        return { status: 'invalid', value: val };
      }
      return { status: 'resolved', value: parsed };
    }
    return { status: 'unset' };
  }

  /**
   * Human-readable dimension description for error messages.
   */
  static describeDim(d: ShapeDimension): string {
    switch (d.kind) {
      case "const":
        return String(d.value);
      case "symbolic":
        return `$${d.name}`;
      case "param_ref":
        return `params.${d.name}`;
      case "wildcard":
        return "*";
      case "computed":
        return `computed(${d.formula})`;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers (module-level, not exported)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check if two dimensions are equal (same kind + same value/name).
 */
function dimEqual(a: ShapeDimension, b: ShapeDimension): boolean {
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "const":
      return a.value === (b as typeof a).value;
    case "symbolic":
      return a.name === (b as typeof a).name;
    case "param_ref":
      return a.name === (b as typeof a).name;
    case "wildcard":
      return true;
    case "computed":
      return a.formula === (b as typeof a).formula && 
             JSON.stringify(a.args) === JSON.stringify((b as typeof a).args);
  }
}

/**
 * Type guard: is the value a TensorType (has `shape` property)?
 */
function isTensorType(
  value: TensorType | TypeError,
): value is TensorType {
  return "shape" in value;
}

/**
 * Type guard: is the value a TypeError?
 */
function isTypeError(
  value: PatternMatchResult | TypeError,
): value is TypeError {
  return !("success" in value);
}
