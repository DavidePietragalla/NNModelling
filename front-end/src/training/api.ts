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

export interface TrainingJobLogs {
  stdout: string;
  stderr: string;
}

export interface TrainingLogTail {
  stdout: TrainingLogChunk;
  stderr: TrainingLogChunk;
}

export interface TrainingLogChunk {
  text: string;
  offset: number;
  reset: boolean;
}

export interface TrainingJobRequest {
  schema_version: number;
  network: { format: "nntree"; value: Record<string, unknown> };
  training: Record<string, unknown>;
  resources: Record<string, unknown>;
  priority: number;
}

export interface PairingGrant {
  request_id: string;
  connection_id: string;
  token: string;
  verification_code: string;
  expires_at: string;
}

export interface PairingStatus {
  request_id: string;
  connection_id: string;
  status: "pending" | "approved" | "rejected" | "expired";
  verification_code: string;
  expires_at: string;
  session_expires_at: string | null;
}

export interface SessionInfo {
  id: string;
  device_name: string | null;
  status: string;
  created_at: string;
  approved_at: string | null;
  expires_at: string | null;
  last_seen_at: string | null;
  revoked_at: string | null;
}

interface ApiErrorBody {
  detail?: string | { code?: string; message?: string };
}

export class BackendApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "BackendApiError";
  }
}

export class TrainingApiClient {
  readonly baseUrl: string;

  constructor(baseUrl: string, private readonly token: string | null = null) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  health(): Promise<{ status: string }> {
    return this.request("/health", {}, false);
  }

  createPairing(deviceName: string | null): Promise<PairingGrant> {
    return this.request("/pairing/requests", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ device_name: deviceName || null }),
    }, false);
  }

  getPairingStatus(requestId: string): Promise<PairingStatus> {
    return this.request(`/pairing/requests/${encodeURIComponent(requestId)}`);
  }

  createRenewal(): Promise<PairingGrant> {
    return this.request("/pairing/renewals", { method: "POST" });
  }

  getSession(): Promise<SessionInfo> {
    return this.request("/session");
  }

  revokeSession(): Promise<SessionInfo> {
    return this.request("/session", { method: "DELETE" });
  }

  listDatasets(): Promise<DatasetInfo[]> {
    return this.request("/datasets");
  }

  listTrainingJobs(): Promise<TrainingJobStatus[]> {
    return this.request("/jobs");
  }

  submitTrainingJob(job: TrainingJobRequest): Promise<TrainingJobStatus> {
    return this.request("/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(job),
    });
  }

  cancelTrainingJob(jobId: string): Promise<TrainingJobStatus> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
  }

  getTrainingJobLogs(jobId: string): Promise<TrainingJobLogs> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}/logs`);
  }

  getTrainingJob(jobId: string): Promise<TrainingJobStatus> {
    return this.request(`/jobs/${encodeURIComponent(jobId)}`);
  }

  tailTrainingJobLogs(jobId: string, stdoutAfter: number, stderrAfter: number): Promise<TrainingLogTail> {
    const query = new URLSearchParams({
      stdout_after: String(stdoutAfter),
      stderr_after: String(stderrAfter),
    });
    return this.request(`/jobs/${encodeURIComponent(jobId)}/logs/tail?${query}`);
  }

  async subscribeTrainingEvents(
    jobId: string,
    onEvent: (event: Record<string, unknown>) => void,
    signal: AbortSignal,
  ): Promise<void> {
    let cursor: string | null = null;
    let terminal = false;
    while (!signal.aborted && !terminal) {
      const headers = this.authHeaders();
      if (cursor) headers.set("last-event-id", cursor);
      const response = await fetch(`${this.baseUrl}/jobs/${encodeURIComponent(jobId)}/events`, {
        headers,
        signal,
      });
      if (!response.ok) throw await responseError(response);
      if (!response.body) throw new Error("Il backend non ha restituito uno stream eventi");
      const parser = new SseParser();
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      while (!signal.aborted) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const message of parser.push(value)) {
          cursor = message.id ?? cursor;
          const event = JSON.parse(message.data) as Record<string, unknown>;
          onEvent(event);
          terminal = ["succeeded", "failed", "cancelled"].includes(String(event.type));
        }
      }
      if (!terminal && !signal.aborted) await abortableDelay(750, signal);
    }
  }

  private async request<T>(path: string, init: RequestInit = {}, authenticated = true): Promise<T> {
    const headers = new Headers(init.headers);
    if (authenticated) {
      for (const [name, value] of this.authHeaders()) headers.set(name, value);
    }
    const response = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) throw await responseError(response);
    return await response.json() as T;
  }

  private authHeaders(): Headers {
    if (!this.token) throw new BackendApiError(401, "missing_token", "La connessione non ha un token");
    return new Headers({ authorization: `Bearer ${this.token}` });
  }
}

export interface SseMessage {
  id: string | null;
  data: string;
}

export class SseParser {
  private buffer = "";
  private id: string | null = null;
  private data: string[] = [];

  push(chunk: string): SseMessage[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const messages: SseMessage[] = [];
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line === "") {
        if (this.data.length > 0) messages.push({ id: this.id, data: this.data.join("\n") });
        this.id = null;
        this.data = [];
      } else if (!line.startsWith(":")) {
        const separator = line.indexOf(":");
        const field = separator >= 0 ? line.slice(0, separator) : line;
        const value = separator >= 0 ? line.slice(separator + 1).replace(/^ /, "") : "";
        if (field === "id") this.id = value;
        if (field === "data") this.data.push(value);
      }
      newline = this.buffer.indexOf("\n");
    }
    return messages;
  }
}

export function canCancelTrainingJob(status: TrainingJobStatus["status"]): boolean {
  return status === "queued" || status === "running";
}

async function responseError(response: Response): Promise<BackendApiError> {
  const body = await response.json().catch(() => undefined) as ApiErrorBody | undefined;
  const detail = body?.detail;
  const code = typeof detail === "object" && detail?.code ? detail.code : `http_${response.status}`;
  const message = typeof detail === "string"
    ? detail
    : typeof detail === "object" && detail?.message
      ? detail.message
      : response.statusText;
  return new BackendApiError(response.status, code, `${response.status}: ${message}`);
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });
}
