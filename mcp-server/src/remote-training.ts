/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 */

/** Thin HTTP client for the optional FastAPI remote-training backend. */

export class RemoteTrainingClient {
  readonly baseUrl: string;

  constructor(baseUrl = process.env.NNM_BACKEND_URL ?? "http://127.0.0.1:8000") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, init);
    const text = await response.text();
    let body: unknown = undefined;
    try {
      body = text ? JSON.parse(text) : undefined;
    } catch {
      body = text;
    }
    if (!response.ok) {
      const detail = typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : text || response.statusText;
      throw new Error(`Training backend ${response.status}: ${detail}`);
    }
    return body as T;
  }

  health(): Promise<Record<string, unknown>> {
    return this.request("/health");
  }

  listDatasets(): Promise<unknown[]> {
    return this.request("/datasets");
  }

  listComputeUnits(): Promise<unknown[]> {
    return this.request("/compute-units");
  }

  submitJob(job: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.request("/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(job),
    });
  }

  getJob(jobId: string): Promise<Record<string, unknown>> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}`);
  }

  listJobs(): Promise<unknown[]> {
    return this.request("/jobs");
  }

  getLogs(jobId: string): Promise<Record<string, string>> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}/logs`);
  }

  cancelJob(jobId: string): Promise<Record<string, unknown>> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
  }

  async getEvents(jobId: string, after?: string): Promise<unknown[]> {
    const cursor = after ? `?after=${encodeURIComponent(after)}` : "";
    const response = await fetch(
      `${this.baseUrl}/jobs/${encodeURIComponent(jobId)}/events${cursor}`,
      { headers: { accept: "text/event-stream" } },
    );
    if (!response.ok) {
      throw new Error(`Training backend ${response.status}: ${response.statusText}`);
    }
    const text = await response.text();
    return text
      .split("\n\n")
      .map((chunk) => chunk.match(/^data: (.+)$/m)?.[1])
      .filter((value): value is string => Boolean(value))
      .map((value) => JSON.parse(value) as unknown);
  }
}
