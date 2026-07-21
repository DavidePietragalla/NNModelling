import { describe, expect, it, vi } from "vitest";
import type { ServerContext } from "../src/server";
import { RemoteTrainingClient } from "../src/remote-training";
import * as remoteTools from "../src/tools/remote-training";

function contextWithClient(): { ctx: ServerContext; client: Record<string, ReturnType<typeof vi.fn>> } {
  const client = {
    listDatasets: vi.fn().mockResolvedValue([{ target: "dataset.test.Dataset" }]),
    listComputeUnits: vi.fn().mockResolvedValue([{ id: "local" }]),
    submitJob: vi.fn().mockResolvedValue({ id: "job-1", status: "queued" }),
    listJobs: vi.fn().mockResolvedValue([]),
    getJob: vi.fn().mockResolvedValue({ id: "job-1" }),
    getLogs: vi.fn().mockResolvedValue({ stdout: "", stderr: "" }),
    getEvents: vi.fn().mockResolvedValue([]),
    cancelJob: vi.fn().mockResolvedValue({ id: "job-1", status: "cancelled" }),
  };
  const ctx = { remoteTraining: client as unknown as RemoteTrainingClient } as ServerContext;
  return { ctx, client };
}

describe("remote training MCP tools", () => {
  it("delegates complete job submission to FastAPI client", async () => {
    const { ctx, client } = contextWithClient();
    const job = { network: {}, training: {}, resources: {}, priority: 0 };

    const result = await remoteTools.submit_training_job.handler(ctx, { job });

    expect(result).toEqual({ id: "job-1", status: "queued" });
    expect(client.submitJob).toHaveBeenCalledWith(job);
  });

  it("delegates dataset, status, logs, events and cancellation tools", async () => {
    const { ctx, client } = contextWithClient();

    await remoteTools.list_training_datasets.handler(ctx);
    await remoteTools.list_training_compute_units.handler(ctx);
    await remoteTools.list_training_jobs.handler(ctx);
    await remoteTools.get_training_job.handler(ctx, { jobId: "job-1" });
    await remoteTools.get_training_job_logs.handler(ctx, { jobId: "job-1" });
    await remoteTools.get_training_job_events.handler(ctx, { jobId: "job-1", after: "2-0" });
    await remoteTools.cancel_training_job.handler(ctx, { jobId: "job-1" });

    expect(client.listDatasets).toHaveBeenCalledOnce();
    expect(client.listComputeUnits).toHaveBeenCalledOnce();
    expect(client.listJobs).toHaveBeenCalledOnce();
    expect(client.getJob).toHaveBeenCalledWith("job-1");
    expect(client.getLogs).toHaveBeenCalledWith("job-1");
    expect(client.getEvents).toHaveBeenCalledWith("job-1", "2-0");
    expect(client.cancelJob).toHaveBeenCalledWith("job-1");
  });
});
