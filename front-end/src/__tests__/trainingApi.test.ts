import { afterEach, describe, expect, it, vi } from "vitest";
import { BackendApiError, SseParser, TrainingApiClient, canCancelTrainingJob } from "../training/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("training job actions", () => {
  it("allows cancellation before and during execution", () => {
    expect(canCancelTrainingJob("queued")).toBe(true);
    expect(canCancelTrainingJob("running")).toBe(true);
  });

  it("does not offer cancellation for terminal jobs", () => {
    expect(canCancelTrainingJob("succeeded")).toBe(false);
    expect(canCancelTrainingJob("failed")).toBe(false);
    expect(canCancelTrainingJob("cancelled")).toBe(false);
  });
});

describe("authenticated training API", () => {
  it("sends the bearer token in headers and never in the URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), { status: 200, headers: { "content-type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const api = new TrainingApiClient("http://backend.lan:8000", "very-secret-token");

    await api.listTrainingJobs();

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://backend.lan:8000/jobs");
    expect(url).not.toContain("very-secret-token");
    expect(new Headers(init.headers).get("authorization")).toBe("Bearer very-secret-token");
  });

  it("exposes machine-readable authentication errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: { code: "session_expired", message: "expired" } }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    ));
    const api = new TrainingApiClient("http://backend.lan:8000", "expired-token");

    await expect(api.getSession()).rejects.toMatchObject<Partial<BackendApiError>>({
      status: 401,
      code: "session_expired",
    });
  });
});

describe("SSE parser", () => {
  it("parses split chunks, comments, multiline data, and event IDs", () => {
    const parser = new SseParser();

    expect(parser.push(": keep-alive\n\nid: 12-0\ndata: {\"type\":\"run")).toEqual([]);
    expect(parser.push("ning\",\ndata: \"step\":1}\n\n")).toEqual([
      { id: "12-0", data: "{\"type\":\"running\",\n\"step\":1}" },
    ]);
  });
});
