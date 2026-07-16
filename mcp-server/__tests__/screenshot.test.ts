/**
 * Screenshot tests use a small fake Chromium DevTools endpoint. This exercises
 * both target discovery over HTTP and screenshot capture over WebSocket without
 * requiring a real browser in the test suite.
 */

import { createServer, type Server } from "node:http";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { WebSocketServer } from "ws";
import { captureChromiumScreenshot } from "../src/chromium-screenshot";

const temporaryDirectories: string[] = [];

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Unable to determine fake DevTools port"));
        return;
      }
      resolve(address.port);
    });
    server.on("error", reject);
  });
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("captureChromiumScreenshot", () => {
  it("discovers the frontend target and writes the captured PNG", async () => {
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const methods: string[] = [];
    let port = 0;

    const httpServer = createServer((request, response) => {
      if (request.url !== "/json/list") {
        response.writeHead(404).end();
        return;
      }

      response.setHeader("content-type", "application/json");
      response.setHeader("connection", "close");
      response.end(
        JSON.stringify([
          {
            id: "page-1",
            title: "NNModelling",
            type: "page",
            url: "http://127.0.0.1:5174/",
            webSocketDebuggerUrl: `ws://127.0.0.1:${port}/devtools/page/page-1`,
          },
        ]),
      );
    });
    const webSocketServer = new WebSocketServer({ server: httpServer });
    webSocketServer.on("connection", (socket) => {
      socket.on("message", (raw) => {
        const request = JSON.parse(raw.toString()) as {
          id: number;
          method: string;
        };
        methods.push(request.method);

        if (request.method === "Page.getLayoutMetrics") {
          socket.send(
            JSON.stringify({
              id: request.id,
              result: {
                cssVisualViewport: { clientWidth: 1440, clientHeight: 900 },
                cssContentSize: { width: 1440, height: 1200 },
              },
            }),
          );
        } else if (request.method === "Page.captureScreenshot") {
          socket.send(
            JSON.stringify({
              id: request.id,
              result: { data: png.toString("base64") },
            }),
          );
        } else {
          socket.send(JSON.stringify({ id: request.id, result: {} }));
        }
      });
    });

    port = await listen(httpServer);
    const directory = mkdtempSync(join(tmpdir(), "nnm-screenshot-test-"));
    temporaryDirectories.push(directory);
    const outputPath = join(directory, "diagram.png");

    try {
      const result = await captureChromiumScreenshot({
        devtoolsUrl: `http://127.0.0.1:${port}`,
        pageUrl: "http://127.0.0.1:5174",
        outputPath,
        hoverNodeId: "linear-1",
      });

      expect(result).toMatchObject({
        success: true,
        outputPath,
        pageId: "page-1",
        pageUrl: "http://127.0.0.1:5174/",
        title: "NNModelling",
        width: 1440,
        height: 900,
        bytes: png.byteLength,
      });
      expect(readFileSync(outputPath)).toEqual(png);
      expect(methods).toEqual([
        "Page.enable",
        "Runtime.enable",
        "Runtime.evaluate",
        "Page.getLayoutMetrics",
        "Page.captureScreenshot",
      ]);
    } finally {
      await new Promise<void>((resolve) => webSocketServer.close(() => resolve()));
      await closeServer(httpServer);
    }
  });
});
