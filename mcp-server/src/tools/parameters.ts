/**
 * Parameter Manipulation Tools — get, set, update, and reset node parameters.
 *
 * Parameters are stored as flat key-value string pairs on node.data.params.
 * The tools validate that:
 *   1. The target node exists
 *   2. The parameter key exists in the stereotype definition (if known)
 *   3. The value matches the expected type (int, float, etc.)
 *
 * Every mutating tool (set, update, reset) pushes a history snapshot
 * before making changes so that parameter edits are undoable.
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import { NodeNotFoundError, ParameterTypeMismatchError } from "../errors";

// ── Helpers ───────────────────────────────────────────────────────────

/**
 * Validate that a parameter value matches its expected type.
 * Types come from the stereotype's ModuleParameter.type field.
 *
 * @param nodeId - Node ID (used in error messages for context).
 * @param key - Parameter name.
 * @param value - Value to validate.
 * @param expectedType - Expected type from stereotype definition.
 * @throws {ParameterTypeMismatchError} If the value cannot be parsed to the expected type.
 */
function validateParamType(nodeId: string, key: string, value: string, expectedType: string): void {
  switch (expectedType) {
    case "int":
    case "integer": {
      const intVal = parseInt(value, 10);
      if (isNaN(intVal) || String(intVal) !== value) {
        throw new ParameterTypeMismatchError(nodeId, key, "int", value);
      }
      break;
    }
    case "float":
    case "number": {
      const floatVal = parseFloat(value);
      if (isNaN(floatVal)) {
        throw new ParameterTypeMismatchError(nodeId, key, "float", value);
      }
      break;
    }
    case "bool":
    case "boolean": {
      if (value !== "true" && value !== "false") {
        throw new ParameterTypeMismatchError(nodeId, key, "bool", value);
      }
      break;
    }
    // "string" and other types accept any value
  }
}

/**
 * Look up the stereotype for a given node by reading its data.stereotype field.
 */
function getNodeStereotype(
  ctx: ServerContext,
  nodeId: string
): { name: string; parameters: Record<string, { type: string; default: string; position?: string }> } | null {
  const node = ctx.diagram.getNodeById(nodeId);
  if (!node) return null;

  const stereoName = (node.data as Record<string, unknown>)?.stereotype as string | undefined;
  if (!stereoName) return null;

  const stereo = ctx.diagram.getStereotype(stereoName);
  if (!stereo) return null;

  return {
    name: stereo.name,
    parameters: stereo.parameters,
  };
}

// ── Schemas ────────────────────────────────────────────────────────────

export const set_parameter = {
  schema: z.object({
    nodeId: z.string().min(1),
    key: z.string().min(1),
    value: z.string(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ nodeId: string; key: string; previousValue: string | null; currentValue: string }> {
    const { nodeId, key, value } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // If we can find the stereotype, validate the parameter key and type
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    if (stereoInfo) {
      const paramDef = stereoInfo.parameters[key];
      if (!paramDef) {
        // Key not in stereotype definition — allow it but note it's a custom param
        // (don't throw, some params might be dynamic)
      } else {
        validateParamType(nodeId, key, value, paramDef.type);
      }
    }

    // Capture previous value before mutation
    const currentParams: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};
    const previousValue = (currentParams[key] as string | undefined) ?? null;

    // Push history snapshot so param changes are undoable
    ctx.history.pushSnapshot("set_parameter", ctx.diagram);

    // Merge with new value
    const updatedParams = {
      ...currentParams,
      [key]: value,
    };

    ctx.diagram.updateModule(nodeId, { params: updatedParams });

    return { nodeId, key, previousValue, currentValue: value };
  },
};

export const update_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
    params: z.record(z.string(), z.string()),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    nodeId: string;
    updated: Array<{ key: string; previousValue: string; currentValue: string }>;
    unchanged: string[];
  }> {
    const { nodeId, params } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // Validate parameter types against stereotype if available
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    if (stereoInfo) {
      for (const [key, value] of Object.entries(params)) {
        const paramDef = stereoInfo.parameters[key];
        if (paramDef) {
          validateParamType(nodeId, key, value, paramDef.type);
        }
      }
    }

    // Read current params and compute changed vs unchanged
    const currentParams: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

    const updated: Array<{ key: string; previousValue: string; currentValue: string }> = [];
    const unchanged: string[] = [];

    for (const [key, value] of Object.entries(params)) {
      const prev = (currentParams[key] as string) ?? "";
      if (prev === value) {
        unchanged.push(key);
      } else {
        updated.push({ key, previousValue: prev, currentValue: value });
      }
    }

    // Push history snapshot so param changes are undoable
    ctx.history.pushSnapshot("update_parameters", ctx.diagram);

    // Merge with batch update
    const updatedParams = {
      ...currentParams,
      ...params,
    };

    ctx.diagram.updateModule(nodeId, { params: updatedParams });

    return { nodeId, updated, unchanged };
  },
};

export const reset_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
    keys: z.array(z.string().min(1)).optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    nodeId: string;
    reset: Array<{ key: string; previousValue: string; defaultValue: string }>;
  }> {
    const { nodeId, keys } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // Get stereotype defaults
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    if (!stereoInfo) {
      // No stereotype found — clear all params to empty
      const currentParams: Record<string, unknown> =
        ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

      const resetEntries = Object.keys(currentParams).map((k) => ({
        key: k,
        previousValue: (currentParams[k] as string) ?? "",
        defaultValue: "",
      }));

      ctx.history.pushSnapshot("reset_parameters", ctx.diagram);
      ctx.diagram.updateModule(nodeId, { params: {} });

      return { nodeId, reset: keys ? resetEntries.filter((r) => keys.includes(r.key)) : resetEntries };
    }

    // Read current params before mutation
    const currentParams: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

    // Determine which keys to reset
    const keysToReset = keys ?? Object.keys(stereoInfo.parameters);
    const reset: Array<{ key: string; previousValue: string; defaultValue: string }> = [];

    const updatedParams = { ...currentParams };
    for (const key of keysToReset) {
      const paramDef = stereoInfo.parameters[key];
      if (!paramDef) continue;
      const prevValue = (currentParams[key] as string) ?? paramDef.default;
      reset.push({ key, previousValue: prevValue, defaultValue: paramDef.default });
      updatedParams[key] = paramDef.default;
    }

    ctx.history.pushSnapshot("reset_parameters", ctx.diagram);
    ctx.diagram.updateModule(nodeId, { params: updatedParams });

    return { nodeId, reset };
  },
};

export const query_parameters = {
  schema: z.object({
    nodeId: z.union([z.string().min(1), z.array(z.string().min(1)).min(1)]),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    nodes: Array<{
      nodeId: string;
      name: string;
      stereotype: string;
      params: Array<{
        key: string;
        value: string;
        type: string;
        default: string;
        position?: "top" | "bottom";
        isModified: boolean;
      }>;
    }>;
  }> {
    // Normalize to array
    const nodeIds = Array.isArray(input.nodeId) ? input.nodeId : [input.nodeId];

    const nodes = nodeIds.map((nid: string) => {
      const node = ctx.diagram.getNodeById(nid);
      if (!node) throw new NodeNotFoundError(nid);

      const nodeData = node.data as Record<string, unknown> | undefined;
      const name = (nodeData?.name as string) ?? nid;
      const stereotype = (nodeData?.stereotype as string) ?? "unknown";

      // Current params from the node (always string values in our system)
      const currentRaw: Record<string, unknown> =
        (nodeData?.params as Record<string, unknown>) ?? {};
      const current = Object.fromEntries(
        Object.entries(currentRaw).map(([k, v]) => [k, String(v ?? "")])
      );

      // Defaults and schema from stereotype
      const stereoInfo = getNodeStereotype(ctx, nid);
      const paramDefs = stereoInfo?.parameters ?? {};

      const params = Object.entries(paramDefs).map(([key, paramDef]) => {
        const currentValue = current[key] ?? paramDef.default;
        return {
          key,
          value: currentValue,
          type: paramDef.type,
          default: paramDef.default,
          position: paramDef.position as "top" | "bottom" | undefined,
          isModified: currentValue !== paramDef.default,
        };
      });

      // Also include any custom params (present in node but not in stereotype)
      for (const [key, value] of Object.entries(current)) {
        if (!paramDefs[key]) {
          params.push({
            key,
            value,
            type: "string",
            default: "",
            position: undefined,
            isModified: true,
          });
        }
      }

      return { nodeId: nid, name, stereotype, params };
    });

    return { nodes };
  },
};
