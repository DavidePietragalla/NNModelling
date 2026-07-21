import { describe, expect, it } from "vitest";
import {
  forgetBackendConnection,
  loadBackendConnection,
  normalizeBackendUrl,
  saveBackendConnection,
  type ConnectionStorage,
  type SavedBackendConnection,
} from "../training/connection";

class MemoryStorage implements ConnectionStorage {
  private values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

describe("training backend connection storage", () => {
  it("normalizes absolute HTTP backend URLs", () => {
    expect(normalizeBackendUrl(" http://192.168.1.20:8000/ ")).toBe("http://192.168.1.20:8000");
    expect(normalizeBackendUrl("https://training.lan/api/")).toBe("https://training.lan/api");
  });

  it.each(["", "/api", "ws://server:8000", "file:///tmp/backend"])(
    "rejects unsupported backend URL %s",
    (value) => expect(() => normalizeBackendUrl(value)).toThrow(),
  );

  it("persists connections by backend URL and restores the active one", () => {
    const storage = new MemoryStorage();
    const first: SavedBackendConnection = {
      version: 1,
      baseUrl: "http://server-a:8000",
      token: "token-a",
      connectionId: "connection-a",
      requestId: null,
      verificationCode: null,
      deviceName: "Laptop",
    };
    const second = { ...first, baseUrl: "http://server-b:8000", token: "token-b" };

    saveBackendConnection(first, storage);
    saveBackendConnection(second, storage);

    expect(loadBackendConnection(storage)).toEqual(second);
    forgetBackendConnection(second.baseUrl, storage);
    expect(loadBackendConnection(storage)).toEqual(first);
  });

  it("ignores malformed or unsupported saved state", () => {
    const storage = new MemoryStorage();
    storage.setItem("nnm.training.connections", JSON.stringify({ version: 999 }));
    expect(loadBackendConnection(storage)).toBeNull();
  });
});
