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

      // ── Step b: No type signature ─────────────────────────────────
      if (!stereotype || !stereotype.typeSignature) {
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
        inputTypes = collected.length > 0 ? collected : undefined;
        // For Phase 1, join inference is a TODO; pass undefined to inferNode
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

      // ── Join kind (Phase 3 TODO) ──────────────────────────────
      case "join": {
        // TODO: implement join type inference (Phase 3)
        return { shape: [], dtype: "unknown" } satisfies TensorType;
      }

      // ── Subflow kind (Phase 4 TODO) ───────────────────────────
      case "subflow": {
        // TODO: implement subflow type inference (Phase 4)
        return { shape: [], dtype: "unknown" } satisfies TensorType;
      }

      default:
        return {
          nodeId: "",
          message: `Unknown type signature kind "${(sig as TypeSignature).kind}" for "${stereotype.name}"`,
          severity: "warning",
        } satisfies TypeError;
    }
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
          if (resolved === undefined) {
            // Param is "Undefined"/missing — treat as symbolic
            captured.push({ kind: "symbolic", name: `?${p.name}` });
            i++;
            j++;
          } else {
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
              paramDim.value !== resolved
            ) {
              return {
                nodeId: "",
                message: `dimension mismatch: param ${p.name}=${resolved}, got ${this.describeDim(paramDim)}`,
                severity: "error",
              } satisfies TypeError;
            }
            i++;
            j++;
          }
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
          if (val !== undefined) {
            result.push({ kind: "const", value: val });
          } else {
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

        default:
          // Unknown pattern kind — keep as unknown symbolic
          result.push({
            kind: "symbolic",
            name: `?unknown_${(p as ShapeDimPattern).kind}`,
          });
          break;
      }
    }

    // Safety: push any unconsumed captured dims (shouldn't happen with valid patterns)
    while (capIdx < captured.length) {
      result.push(captured[capIdx]);
      capIdx++;
    }

    return result;
  }

  /**
   * Resolve a parameter reference from a node's parameter map.
   *
   * Handles both flat values (`params[name] === "784"`) and wrapped objects
   * (`params[name] === { value: "784", position: "top" }`).
   *
   * Returns undefined if the parameter is "Undefined", missing, or cannot be
   * parsed as a number.
   */
  static resolveParamRef(
    name: string,
    params: Record<string, unknown>,
  ): number | undefined {
    const raw = params[name];
    if (raw === undefined || raw === null) return undefined;

    // Handle wrapped parameter objects: { value: "784", position: "top" }
    const val: unknown =
      typeof raw === "object" && raw !== null && "value" in raw
        ? (raw as Record<string, unknown>).value
        : raw;

    if (typeof val === "number") return val;
    if (typeof val === "string") {
      if (
        val === "Undefined" ||
        val === "" ||
        val === "None" ||
        val === "True" ||
        val === "False"
      ) {
        return undefined;
      }
      const parsed = Number(val);
      return isNaN(parsed) ? undefined : parsed;
    }
    return undefined;
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
