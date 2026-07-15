/**
 * @file Tensor type system — core type definitions.
 *
 * This file contains ONLY interfaces and type aliases.  No runtime logic.
 * No imports from other project files.  The types here are consumed by the
 * type inference engine (typeEngine.ts) and by stereotype declarations.
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1. ShapeDimension — resolved single dimension
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single dimension in a tensor shape.  After inference all symbolic and
 * param_ref names have been resolved or unified, leaving only const and
 * wildcard dimensions.
 *
 * - `'const'` — a literal integer (e.g. 784, 64, 10).
 * - `'symbolic'` — a named variable (e.g. batch size `B`, sequence length `S`).
 * - `'param_ref'` — references a node parameter (e.g. `in_features`).
 * - `'wildcard'` — matches zero or more arbitrary trailing dimensions.
 * - `'computed'` — a dimension computed by a formula (e.g. conv2d_hw).
 *   `value` is set when formula arguments have been fully resolved.
 */
export type ShapeDimension =
  | { kind: 'const'; value: number }
  | { kind: 'symbolic'; name: string }
  | { kind: 'param_ref'; name: string }
  | { kind: 'wildcard' }
  | { kind: 'computed'; expr?: string; formula?: string; args?: string[]; value?: number };

// ─────────────────────────────────────────────────────────────────────────────
// 2. TensorShape — ordered list of dimensions
// ─────────────────────────────────────────────────────────────────────────────

/** Ordered list of dimensions forming a tensor shape. */
export type TensorShape = ShapeDimension[];

// ─────────────────────────────────────────────────────────────────────────────
// 3. DType — data type (string-based, extensible)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Data type of a tensor.  Expressed as a plain string so that new dtypes can
 * be introduced without modifying this file.  Common values:
 * `'float32'`, `'float64'`, `'int64'`, `'int32'`, `'bool'`, `'unknown'`.
 */
export type DType = string;

// ─────────────────────────────────────────────────────────────────────────────
// 4. TensorType — shape + dtype
// ─────────────────────────────────────────────────────────────────────────────

/** Fully-resolved type of a tensor flowing between nodes. */
export interface TensorType {
  shape: TensorShape;
  dtype: DType;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ShapeDimPattern — unresolved shape dimension (declarative form)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A dimension pattern as it appears in a stereotype's `type_signature` JSON.
 * Parallels `ShapeDimension` but represents *unresolved* patterns —
 * symbolic names (e.g. `$B`) and param refs (e.g. `params.in_features`)
 * have not yet been resolved.
 *
 * The leading `$` from JSON symbolic names is stripped on load, so
 * `"$B"` becomes `{ kind: 'symbolic', name: 'B' }`.
 *
 * `'computed'` — a dimension whose size is determined by a named formula
 * (e.g. `conv2d_hw`) with arguments that reference symbolic dims (`$H`)
 * or node params (`kernel_size`).  Computed dimensions are resolved during
 * `resolvePattern` when all arguments are known.
 */
export type ShapeDimPattern =
  | { kind: 'const'; value: number }
  | { kind: 'symbolic'; name: string }
  | { kind: 'param_ref'; name: string }
  | { kind: 'wildcard' }
  | { kind: 'computed'; expr?: string; formula?: string; args?: string[] };

// ─────────────────────────────────────────────────────────────────────────────
// 6. ShapePattern — ordered list of dimension patterns
// ─────────────────────────────────────────────────────────────────────────────

/** Ordered list of dimension patterns forming an unresolved shape. */
export type ShapePattern = ShapeDimPattern[];

// ─────────────────────────────────────────────────────────────────────────────
// 7. TypeSignature — stereotype-declared type signature
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Type signature declared by a stereotype JSON (field `type_signature`).
 *
 * - `kind`: discriminates between standard modules, join nodes, and subflows.
 * - `input`: shape pattern(s) accepted.  A single `ShapePattern` for modules
 *   and subflows; an array of `ShapePattern` for joins (one per input handle).
 * - `output`: shape pattern the node produces.
 * - `dtype`: optional dtype constraints on inputs and/or output.
 * - `constraints`: additional constraints on the type relationship between inputs.
 *   Currently supports:
 *   - `concat`: for Concat joins, specifies which dimension is concatenated.
 *     `dim` is a string like `"params.dim"` resolving to a parameter on the node.
 */
export interface TypeSignature {
  /** 'module' | 'join' | 'subflow' */
  kind: 'module' | 'join' | 'subflow';

  /** Input pattern(s). Single ShapePattern for modules, array for joins. */
  input: ShapePattern | ShapePattern[];

  /** Output pattern. */
  output: ShapePattern;

  /** Optional dtype constraints. */
  dtype?: {
    input?: DType;
    output?: DType;
  };

  /** Optional additional constraints (e.g. concat dimension, hrepeat). */
  constraints?: {
    concat?: { dim: string };
    hrepeat?: { n: string };
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. TypeError — error/warning during type inference
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error or warning produced during type inference, associated with a specific
 * node in the diagram.
 */
export interface TypeError {
  /** ID of the node that caused the error. */
  nodeId: string;

  /** Human-readable description of the problem. */
  message: string;

  /** `'error'` indicates the diagram cannot be converted; `'warning'` is informational. */
  severity: 'error' | 'warning';
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. NodeTypeAnnotation — per-node type information
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Result of type inference for a single node: what types enter and leave.
 *
 * - `inputType`: type arriving from the single upstream connection
 *   (undefined for source/Input nodes that have no incoming edge).
 * - `inputTypes`: for join nodes, one type per input handle
 *   (in-0, in-1, …).  Undefined for non-join nodes.
 * - `outputType`: type produced by this node.
 */
export interface NodeTypeAnnotation {
  /** ID of the node this annotation belongs to. */
  nodeId: string;

  /** Type arriving at this node (undefined for Input/source nodes). */
  inputType?: TensorType;

  /** For join nodes: one type per input handle (in-0, in-1, …). */
  inputTypes?: TensorType[];

  /** Type produced by this node. */
  outputType: TensorType;
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. TypeResult — complete inference output
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Complete result of a type inference run over a diagram.
 *
 * - `ok`: whether inference completed without errors.
 * - `annotations`: map from node ID to the inferred type annotation.
 * - `errors`: list of errors or warnings that were encountered.
 */
export interface TypeResult {
  ok: boolean;
  annotations: Map<string, NodeTypeAnnotation>;
  errors: TypeError[];
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. TypeEnvironment — symbolic name bindings
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Bindings of symbolic dimension names to concrete (or still-symbolic)
 * dimensions, built up during type inference.
 *
 * Keys are the symbolic name (e.g. `"B"`, `"S"`); values are the
 * `ShapeDimension` they have been unified with.
 */
export type TypeEnvironment = Map<string, ShapeDimension>;

// ─────────────────────────────────────────────────────────────────────────────
// 12. ParamResolution — result of resolving a parameter reference
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Discriminated union for parameter resolution results.
 *
 * - `'unset'` — parameter is "Undefined", "", or "None" (user hasn't set it yet).
 * - `'invalid'` — parameter has a value but it cannot be parsed as a number
 *   (e.g. "cazz", "hello"). This is a type error.
 * - `'resolved'` — parameter resolved to a concrete numeric value.
 */
export type ParamResolution =
  | { status: 'unset' }
  | { status: 'invalid'; value: string }
  | { status: 'resolved'; value: number };
