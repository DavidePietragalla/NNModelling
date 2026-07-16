/**
 * Process lifecycle regression tests for the MCP stdio executable.
 *
 * The WebSocket listener must not outlive its MCP client. In particular,
 * closing stdin is the normal way a stdio client disconnects and must release
 * the configured port before the process exits.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { BrowserRPCClient } from "../src/browser-client";

function findAvailablePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = require("node:net").createServer();
    server.listen(0, "localhost", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Unable to determine test port"));
        return;
      }

      server.close(() => resolve(address.port));
    });
    server.on("error", reject);
  });
}

function waitForStderr(
  child: ChildProcessWithoutNullStreams,
  expected: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let output = "";
    const timeout = setTimeout(() => {
      reject(new Error(`Timed out waiting for '${expected}'. Stderr:\n${output}`));
    }, 10_000);

    child.stderr.on("data", (chunk: Buffer) => {
      output += chunk.toString();
      if (output.includes(expected)) {
        clearTimeout(timeout);
        resolve();
      }
    });

    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      reject(
        new Error(
          `MCP process exited before startup (code=${code}, signal=${signal}). Stderr:\n${output}`,
        ),
      );
    });
  });
}

function waitForExit(
  child: ChildProcessWithoutNullStreams,
): Promise<{ code: number | null; signal: NodeJS.Signals | null }> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("MCP process did not exit after stdin closed"));
    }, 10_000);

    child.once("exit", (code, signal) => {
      clearTimeout(timeout);
      resolve({ code, signal });
    });
  });
}

describe("MCP process lifecycle", () => {
  it("releases the WebSocket port when its stdio client disconnects", async () => {
    const port = await findAvailablePort();
    const entrypoint = fileURLToPath(new URL("../dist/index.js", import.meta.url));
    const child = spawn(process.execPath, [entrypoint], {
      env: {
        ...process.env,
        NNM_WS_PORT: String(port),
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    try {
      await waitForStderr(child, "Server connected via stdio");

      const exitPromise = waitForExit(child);
      child.stdin.end();

      await expect(exitPromise).resolves.toEqual({ code: 0, signal: null });

      const replacement = new BrowserRPCClient({ port });
      await expect(replacement.start()).resolves.toBeUndefined();
      await replacement.close();
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
      }
    }
  });
});
