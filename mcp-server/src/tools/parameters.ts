/**
 * Parameter Manipulation Tools — get, set, update, and reset node parameters.
 *
 * Parameters are stored as flat key-value string pairs on node.data.params.
 * The tools validate that:
 *   1. The target node exists
 *   2. The parameter key exists in the stereotype definition (if known)
 *   3. The value matches the expected type (int, float, etc.)
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import { NodeNotFoundError, ParameterNotFoundError, ParameterTypeMismatchError } from "../errors";

// ── Helpers ───────────────────────────────────────────────────────────

/**
 * Validate that a parameter value matches its expected type.
 * Types come from the stereotype's ModuleParameter.type field.
 */
function validateParamType(key: string, value: string, expectedType: string): void {
  switch (expectedType) {
    case "int":
    case "integer": {
      const intVal = parseInt(value, 10);
      if (isNaN(intVal) || String(intVal) !== value) {
        throw new ParameterTypeMismatchError("", key, "int", value);
      }
      break;
    }
    case "float":
    case "number": {
      const floatVal = parseFloat(value);
      if (isNaN(floatVal)) {
        throw new ParameterTypeMismatchError("", key, "float", value);
      }
      break;
    }
    case "bool":
    case "boolean": {
      if (value !== "true" && value !== "false") {
        throw new ParameterTypeMismatchError("", key, "bool", value);
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
  ): Promise<{ nodeId: string; key: string; value: string }> {
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
        validateParamType(key, value, paramDef.type);
      }
    }

    // Read current params, merge with new value
    const currentParams: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

    const updatedParams = {
      ...currentParams,
      [key]: value,
    };

    ctx.diagram.updateModule(nodeId, { params: updatedParams });

    return { nodeId, key, value };
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
  ): Promise<{ nodeId: string; updated: string[] }> {
    const { nodeId, params } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // Validate parameter types against stereotype if available
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    const updatedKeys: string[] = [];

    if (stereoInfo) {
      for (const [key, value] of Object.entries(params)) {
        const paramDef = stereoInfo.parameters[key];
        if (paramDef) {
          validateParamType(key, value, paramDef.type);
        }
        updatedKeys.push(key);
      }
    } else {
      updatedKeys.push(...Object.keys(params));
    }

    // Read current params and merge with batch update
    const currentParams: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

    const updatedParams = {
      ...currentParams,
      ...params,
    };

    ctx.diagram.updateModule(nodeId, { params: updatedParams });

    return { nodeId, updated: updatedKeys };
  },
};

export const reset_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{ nodeId: string; params: Record<string, string> }> {
    const { nodeId } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // Get stereotype defaults
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    if (!stereoInfo) {
      // No stereotype found — clear params to empty
      ctx.diagram.updateModule(nodeId, { params: {} });
      return { nodeId, params: {} };
    }

    // Build default params map (flat key-value strings from defaults)
    const defaults: Record<string, string> = {};
    for (const [key, paramDef] of Object.entries(stereoInfo.parameters)) {
      defaults[key] = paramDef.default;
    }

    ctx.diagram.updateModule(nodeId, { params: defaults });

    return { nodeId, params: defaults };
  },
};

export const query_parameters = {
  schema: z.object({
    nodeId: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>
  ): Promise<{
    nodeId: string;
    current: Record<string, unknown>;
    defaults: Record<string, string>;
    schema: Record<string, { type: string; default: string; position?: string }>;
  }> {
    const { nodeId } = input;

    // Validate node exists
    const node = ctx.diagram.getNodeById(nodeId);
    if (!node) throw new NodeNotFoundError(nodeId);

    // Current params from the node
    const current: Record<string, unknown> =
      ((node.data as Record<string, unknown>)?.params as Record<string, unknown>) ?? {};

    // Defaults and schema from stereotype
    const stereoInfo = getNodeStereotype(ctx, nodeId);
    const defaults: Record<string, string> = {};
    const schema: Record<string, { type: string; default: string; position?: string }> = {};

    if (stereoInfo) {
      for (const [key, paramDef] of Object.entries(stereoInfo.parameters)) {
        defaults[key] = paramDef.default;
        schema[key] = {
          type: paramDef.type,
          default: paramDef.default,
          position: paramDef.position,
        };
      }
    }

    return { nodeId, current, defaults, schema };
  },
};
