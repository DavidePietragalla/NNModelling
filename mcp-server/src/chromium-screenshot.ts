/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * Licensed under the GNU General Public License v3 or later.
 * Commercial licenses are available — contact Luca Sforza.
 * See the LICENSE file for details.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { WebSocket } from "ws";

interface DevToolsTarget {
  id: string;
  title?: string;
  type: string;
  url: string;
  webSocketDebuggerUrl?: string;
}

interface CDPError {
  message: string;
}

interface CDPResponse<T = unknown> {
  id?: number;
  result?: T;
  error?: CDPError;
}

interface PendingCommand {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface LayoutMetrics {
  cssVisualViewport?: {
    clientWidth?: number;
    clientHeight?: number;
  };
  cssContentSize?: {
    width?: number;
    height?: number;
  };
}

interface ScreenshotResponse {
  data?: string;
}

interface RuntimeEvaluateResponse {
  result?: { value?: unknown };
  exceptionDetails?: {
    text?: string;
    exception?: { description?: string };
  };
}

export interface CaptureScreenshotOptions {
  outputPath?: string;
  pageUrl?: string;
  fullPage?: boolean;
  hoverNodeId?: string;
  reloadPage?: boolean;
  devtoolsUrl?: string;
  timeoutMs?: number;
}

export interface CaptureScreenshotResult {
  success: true;
  outputPath: string;
  pageId: string;
  pageUrl: string;
  title: string;
  width: number | null;
  height: number | null;
  bytes: number;
}

class DevToolsProtocolClient {
  private nextId = 0;
  private pending = new Map<number, PendingCommand>();

  private constructor(
    private readonly socket: WebSocket,
    private readonly timeoutMs: number,
  ) {
    socket.on("message", (raw) => this.handleMessage(raw.toString()));
    socket.on("close", () => this.rejectAll(new Error("DevTools connection closed")));
    socket.on("error", (error) => this.rejectAll(error));
  }

  static async connect(
    webSocketUrl: string,
    timeoutMs: number,
  ): Promise<DevToolsProtocolClient> {
    const socket = new WebSocket(webSocketUrl);

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        socket.terminate();
        reject(new Error("Timed out connecting to Chromium DevTools"));
      }, timeoutMs);

      const cleanup = (): void => {
        clearTimeout(timer);
        socket.off("open", onOpen);
        socket.off("error", onError);
      };
      const onOpen = (): void => {
        cleanup();
        resolve();
      };
      const onError = (error: Error): void => {
        cleanup();
        reject(error);
      };

      socket.once("open", onOpen);
      socket.once("error", onError);
    });

    return new DevToolsProtocolClient(socket, timeoutMs);
  }

  call<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("Chromium DevTools connection is not open"));
    }

    const id = ++this.nextId;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`DevTools command timed out: ${method}`));
      }, this.timeoutMs);

      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timer,
      });

      try {
        this.socket.send(JSON.stringify({ id, method, params }));
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error as Error);
      }
    });
  }

  close(): void {
    this.rejectAll(new Error("DevTools client closed"));
    this.socket.terminate();
  }

  private handleMessage(raw: string): void {
    let message: CDPResponse;
    try {
      message = JSON.parse(raw) as CDPResponse;
    } catch {
      return;
    }

    if (message.id === undefined) return;
    const command = this.pending.get(message.id);
    if (!command) return;

    clearTimeout(command.timer);
    this.pending.delete(message.id);

    if (message.error) command.reject(new Error(message.error.message));
    else command.resolve(message.result);
  }

  private rejectAll(error: Error): void {
    for (const command of this.pending.values()) {
      clearTimeout(command.timer);
      command.reject(error);
    }
    this.pending.clear();
  }
}

function normalizeDevToolsUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

function isLocalFrontend(target: DevToolsTarget): boolean {
  try {
    const hostname = new URL(target.url).hostname;
    return hostname === "localhost" || hostname === "127.0.0.1";
  } catch {
    return false;
  }
}

function selectTarget(
  targets: DevToolsTarget[],
  preferredUrl?: string,
): DevToolsTarget {
  const pages = targets.filter(
    (target) => target.type === "page" && target.webSocketDebuggerUrl,
  );

  if (preferredUrl) {
    const preferred = pages.find((target) => target.url.startsWith(preferredUrl));
    if (!preferred) {
      throw new Error(
        `No Chromium page matches '${preferredUrl}'. Available pages: ` +
          (pages.map((target) => target.url).join(", ") || "none"),
      );
    }
    return preferred;
  }

  const localPages = pages.filter(isLocalFrontend);
  if (localPages.length === 1) return localPages[0];
  if (pages.length === 1) return pages[0];

  throw new Error(
    "Unable to select the NNModelling page unambiguously. " +
      "Pass pageUrl or set NNM_FRONTEND_URL. Available pages: " +
      (pages.map((target) => target.url).join(", ") || "none"),
  );
}

async function fetchTargets(
  devtoolsUrl: string,
  timeoutMs: number,
): Promise<DevToolsTarget[]> {
  let response: Response;
  try {
    response = await fetch(`${normalizeDevToolsUrl(devtoolsUrl)}/json/list`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    throw new Error(
      `Unable to connect to Chromium DevTools at ${devtoolsUrl}. ` +
        "Start Chromium with --remote-debugging-port=9223. " +
        `Cause: ${(error as Error).message}`,
    );
  }

  if (!response.ok) {
    throw new Error(
      `Chromium DevTools returned HTTP ${response.status} at ${devtoolsUrl}`,
    );
  }

  const targets = await response.json();
  if (!Array.isArray(targets)) {
    throw new Error("Chromium DevTools returned an invalid target list");
  }
  return targets as DevToolsTarget[];
}

export async function captureChromiumScreenshot(
  options: CaptureScreenshotOptions = {},
): Promise<CaptureScreenshotResult> {
  const timeoutMs = options.timeoutMs ?? 5_000;
  const devtoolsUrl =
    options.devtoolsUrl ?? process.env.NNM_CDP_URL ?? "http://127.0.0.1:9223";
  const pageUrl = options.pageUrl ?? process.env.NNM_FRONTEND_URL;
  const targets = await fetchTargets(devtoolsUrl, timeoutMs);
  const target = selectTarget(targets, pageUrl);
  const client = await DevToolsProtocolClient.connect(
    target.webSocketDebuggerUrl!,
    timeoutMs,
  );

  try {
    await client.call("Page.enable");
    if (options.reloadPage) {
      await client.call("Page.reload", { ignoreCache: true });
      const deadline = Date.now() + timeoutMs;
      let loaded = false;
      while (Date.now() < deadline) {
        try {
          const state = await client.call<RuntimeEvaluateResponse>(
            "Runtime.evaluate",
            {
              expression: "document.readyState",
              returnByValue: true,
            },
          );
          if (state.result?.value === "complete") {
            loaded = true;
            break;
          }
        } catch {
          // The execution context is briefly unavailable during navigation.
        }
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      if (!loaded) throw new Error("Timed out waiting for the page to reload");
    }
    if (options.hoverNodeId) {
      await client.call("Runtime.enable");
      const nodeId = JSON.stringify(options.hoverNodeId);
      const evaluation = await client.call<RuntimeEvaluateResponse>(
        "Runtime.evaluate",
        {
          expression: `
            (async () => {
              const nodeId = ${nodeId};
              const node = Array.from(document.querySelectorAll("[data-id]"))
                .find((element) => element.getAttribute("data-id") === nodeId);
              const handle = node?.querySelector(".output-handle-wrapper");
              if (!handle) throw new Error("No output handle found for node " + nodeId);
              handle.dispatchEvent(new MouseEvent("mouseenter"));
              await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
              await new Promise((resolve) => setTimeout(resolve, 150));
              return true;
            })()
          `,
          awaitPromise: true,
          returnByValue: true,
        },
      );
      if (evaluation.exceptionDetails) {
        throw new Error(
          evaluation.exceptionDetails.exception?.description ??
            evaluation.exceptionDetails.text ??
            `Unable to hover node '${options.hoverNodeId}'`,
        );
      }
    }
    const metrics = await client.call<LayoutMetrics>("Page.getLayoutMetrics");
    const screenshot = await client.call<ScreenshotResponse>(
      "Page.captureScreenshot",
      {
        format: "png",
        fromSurface: true,
        captureBeyondViewport: options.fullPage ?? false,
      },
    );

    if (!screenshot.data) {
      throw new Error("Chromium DevTools returned an empty screenshot");
    }

    const buffer = Buffer.from(screenshot.data, "base64");
    const outputPath = resolve(
      options.outputPath ??
        join(tmpdir(), `nnmodelling-screenshot-${Date.now()}.png`),
    );
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, buffer);

    const width = options.fullPage
      ? metrics.cssContentSize?.width
      : metrics.cssVisualViewport?.clientWidth;
    const height = options.fullPage
      ? metrics.cssContentSize?.height
      : metrics.cssVisualViewport?.clientHeight;

    return {
      success: true,
      outputPath,
      pageId: target.id,
      pageUrl: target.url,
      title: target.title ?? "",
      width: width ?? null,
      height: height ?? null,
      bytes: buffer.byteLength,
    };
  } finally {
    client.close();
  }
}
