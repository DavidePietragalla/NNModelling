export interface DatasetParameter {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
}

export interface DatasetInfo {
  target: string;
  name: string;
  doc: string;
  parameters: DatasetParameter[];
}

export interface TrainingJobStatus {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  priority: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  executor: string | null;
  compute_unit: string | null;
  error: string | null;
  heartbeat_at: string | null;
  wandb_url: string | null;
  artifact_dir: string;
}

export interface TrainingJobRequest {
  schema_version: number;
  network: { format: "nntree"; value: Record<string, unknown> };
  training: Record<string, unknown>;
  resources: Record<string, unknown>;
  priority: number;
}

const configuredBase = (import.meta.env.VITE_TRAINING_API_URL as string | undefined) ?? "/api";
const base = configuredBase.replace(/\/$/, "");

export function trainingApiUrl(path: string): string {
  return `${base}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(trainingApiUrl(path), init);
  const body = await response.json().catch(() => undefined);
  if (!response.ok) {
    const detail = body && typeof body === "object" && "detail" in body
      ? String((body as { detail: unknown }).detail)
      : response.statusText;
    throw new Error(`${response.status}: ${detail}`);
  }
  return body as T;
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return request<DatasetInfo[]>("/datasets");
}

export function listTrainingJobs(): Promise<TrainingJobStatus[]> {
  return request<TrainingJobStatus[]>("/jobs");
}

export function submitTrainingJob(job: TrainingJobRequest): Promise<TrainingJobStatus> {
  return request<TrainingJobStatus>("/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(job),
  });
}

export function cancelTrainingJob(jobId: string): Promise<TrainingJobStatus> {
  return request<TrainingJobStatus>(`/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}

export function trainingEventsUrl(jobId: string): string {
  return trainingApiUrl(`/jobs/${encodeURIComponent(jobId)}/events`);
}

