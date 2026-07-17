/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 */

/** Optional MCP proxy tools for the FastAPI remote-training backend. */

import { z } from "zod";
import type { ServerContext } from "../server.js";
import { RemoteTrainingClient } from "../remote-training.js";

function client(ctx: ServerContext): RemoteTrainingClient {
  return ctx.remoteTraining ?? new RemoteTrainingClient();
}

export const list_training_datasets = {
  schema: z.object({}),

  async handler(ctx: ServerContext) {
    return client(ctx).listDatasets();
  },
};

export const list_training_compute_units = {
  schema: z.object({}),

  async handler(ctx: ServerContext) {
    return client(ctx).listComputeUnits();
  },
};

export const submit_training_job = {
  schema: z.object({
    job: z.record(z.unknown()).describe("Complete NNModelling training job JSON"),
  }),

  async handler(ctx: ServerContext, input: { job: Record<string, unknown> }) {
    return client(ctx).submitJob(input.job);
  },
};

export const list_training_jobs = {
  schema: z.object({}),

  async handler(ctx: ServerContext) {
    return client(ctx).listJobs();
  },
};

export const get_training_job = {
  schema: z.object({ jobId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: { jobId: string }) {
    return client(ctx).getJob(input.jobId);
  },
};

export const get_training_job_logs = {
  schema: z.object({ jobId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: { jobId: string }) {
    return client(ctx).getLogs(input.jobId);
  },
};

export const get_training_job_events = {
  schema: z.object({ jobId: z.string().min(1), after: z.number().int().nonnegative().optional() }),

  async handler(ctx: ServerContext, input: { jobId: string; after?: number }) {
    return client(ctx).getEvents(input.jobId, input.after ?? 0);
  },
};

export const cancel_training_job = {
  schema: z.object({ jobId: z.string().min(1) }),

  async handler(ctx: ServerContext, input: { jobId: string }) {
    return client(ctx).cancelJob(input.jobId);
  },
};

