/**
 * Compilation, Serialization, and Pipeline Execution Tools.
 *
 * These tools bridge the diagram state (DiagramCore) with the Python
 * training pipeline:
 *   1. Compile the diagram into an NNTree JSON representation
 *   2. Export/import diagram JSON for persistence
 *   3. Execute the Python conversion pipeline (convert.py → Hydra YAML)
 *   4. Execute training (main.py)
 *   5. Execute inference (infer.py)
 *
 * @module tools/conversion
 */

import { z } from "zod";
import type { ServerContext } from "../server";
import { NNTree } from "@nnmodelling/front-end/conversion/nnTree";
import { CompilationFailedError, ImportFailedError, ExportFailedError } from "../errors";
import type {
  ConversionResult,
  TrainingResult,
  InferenceResult,
} from "../pipeline";
import type { NNTreeOutput } from "@nnmodelling/front-end/core/types";

// ── compile_nntree ─────────────────────────────────────────────────────

export const compile_nntree = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<NNTreeOutput> {
    try {
      const nntree = new NNTree(ctx.diagram);
      const json = nntree.toJson();
      const parsed = JSON.parse(json);

      // Count subflows in the compiled tree
      let subflowCount = 0;
      for (const [, node] of nntree.nodes) {
        if (node.isSubflow()) subflowCount++;
      }

      return {
        json,
        root: nntree.root,
        nodeCount: nntree.nodes.size,
        subflowCount,
        lossNodeType: nntree.lossNode?.stereotype ?? null,
      };
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unknown compilation error";
      throw new CompilationFailedError(message);
    }
  },
};

// ── export_diagram ─────────────────────────────────────────────────────

export const export_diagram = {
  schema: z.object({}),

  async handler(
    ctx: ServerContext,
    _input: z.infer<typeof this.schema>,
  ): Promise<{
    json: string;
    nodeCount: number;
    edgeCount: number;
  }> {
    try {
      const json = ctx.diagram.exportToJson();
      return {
        json,
        nodeCount: ctx.diagram.nodes.length,
        edgeCount: ctx.diagram.edges.length,
      };
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unknown export error";
      throw new ExportFailedError(message);
    }
  },
};

// ── import_diagram ─────────────────────────────────────────────────────

export const import_diagram = {
  schema: z.object({
    json: z.string().min(1),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<{
    nodeCount: number;
    edgeCount: number;
  }> {
    // Validate JSON format before importing
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(input.json);
    } catch {
      throw new ImportFailedError("Invalid JSON string — could not parse.");
    }

    if (
      !Array.isArray(parsed.nodes) ||
      !Array.isArray(parsed.edges)
    ) {
      throw new ImportFailedError(
        "JSON must contain both 'nodes' (array) and 'edges' (array) fields.",
      );
    }

    // Push history snapshot before mutation
    ctx.history.pushSnapshot("import_diagram", ctx.diagram);

    try {
      ctx.diagram.importFromJson(input.json);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unknown import error";
      throw new ImportFailedError(message);
    }

    return {
      nodeCount: ctx.diagram.nodes.length,
      edgeCount: ctx.diagram.edges.length,
    };
  },
};

// ── execute_conversion ─────────────────────────────────────────────────

export const execute_conversion = {
  schema: z.object({
    outputDir: z.string().min(1),
    numClasses: z.number().int().positive().optional(),
    dataset: z.string().optional(),
    earlyStopPatience: z.number().int().nonnegative().optional(),
    earlyStopMinDelta: z.number().nonnegative().optional(),
    maxEpochs: z.number().int().positive().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<ConversionResult> {
    // Step 1: Compile NNTree
    const nntreeOutput = await compile_nntree.handler(ctx, {});

    // Step 2: Run Python conversion
    const result = await ctx.pipeline.executeConversion(nntreeOutput.json, {
      outputDir: input.outputDir,
      numClasses: input.numClasses,
      dataset: input.dataset,
      earlyStopPatience: input.earlyStopPatience,
      earlyStopMinDelta: input.earlyStopMinDelta,
      maxEpochs: input.maxEpochs,
    });

    // Step 3: Push history snapshot
    ctx.history.pushSnapshot("execute_conversion", ctx.diagram);

    return result;
  },
};

// ── execute_training ───────────────────────────────────────────────────

export const execute_training = {
  schema: z.object({
    configDir: z.string().min(1),
    configName: z.string().optional(),
    device: z.enum(["cpu", "gpu"]).optional(),
    maxEpochs: z.number().int().positive().optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<TrainingResult> {
    const result = await ctx.pipeline.executeTraining({
      configDir: input.configDir,
      configName: input.configName,
      device: input.device,
      maxEpochs: input.maxEpochs,
    });

    return result;
  },
};

// ── execute_inference ──────────────────────────────────────────────────

export const execute_inference = {
  schema: z.object({
    configDir: z.string().min(1),
    configName: z.string().optional(),
    weightsPath: z.string().min(1),
    outputPath: z.string().optional(),
    imageDir: z.string().optional(),
    device: z.enum(["cpu", "gpu"]).optional(),
  }),

  async handler(
    ctx: ServerContext,
    input: z.infer<typeof this.schema>,
  ): Promise<InferenceResult> {
    const result = await ctx.pipeline.executeInference({
      configDir: input.configDir,
      configName: input.configName,
      weightsPath: input.weightsPath,
      outputPath: input.outputPath,
      imageDir: input.imageDir,
      device: input.device,
    });

    return result;
  },
};
