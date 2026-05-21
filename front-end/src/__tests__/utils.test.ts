import { describe, it, expect } from "vitest";
import { type Edge } from "@xyflow/svelte";
import { checkValidConnection } from "../utils";

// Minimal mock Diagram — only edges property needed by checkValidConnection
class MockDiagram {
  edges: Edge[];
  constructor(edges: Edge[]) {
    this.edges = edges;
  }
}

describe("checkValidConnection", () => {
  it("allows connection when no edges exist", () => {
    const diagram = new MockDiagram([]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });

  it("allows connection when target is different from all existing edges", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "y", targetHandle: "in-0" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });

  it("allows connection when same target but different handle", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "b", targetHandle: "in-0" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-1" };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });

  it("blocks connection when same target + same targetHandle is taken", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "b", targetHandle: "in-0" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(diagram as any, conn)).toBe(false);
  });

  it("blocks when passed an existing Edge instead of Connection", () => {
    const existingEdge: Edge = {
      id: "e1",
      source: "x",
      target: "b",
      targetHandle: "in-0",
    };
    const diagram = new MockDiagram([existingEdge]);
    // Trying to create an edge that matches the existing one
    expect(checkValidConnection(diagram as any, existingEdge)).toBe(false);
  });

  it("allows connection when target handle is undefined and free", () => {
    const diagram = new MockDiagram([]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: null };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });

  it("blocks connection when target handle is undefined but another edge with undefined already exists", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "b" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: null };
    expect(checkValidConnection(diagram as any, conn)).toBe(false);
  });

  it("allows connection to join node with multiple inputs when specific free handle exists", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "join1", targetHandle: "in-0" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "join1", targetHandle: "in-1" };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });

  it("blocks connection to join node when specific handle is already used", () => {
    const diagram = new MockDiagram([
      { id: "e1", source: "x", target: "join1", targetHandle: "in-0" },
    ]);
    const conn = { source: "a", sourceHandle: null, target: "join1", targetHandle: "in-0" };
    expect(checkValidConnection(diagram as any, conn)).toBe(false);
  });

  it("allows source handle duplicates (source handles not checked)", () => {
    // Source handles can have unlimited outgoing connections
    const diagram = new MockDiagram([
      { id: "e1", source: "a", sourceHandle: "out", target: "b", targetHandle: "in-0" },
    ]);
    // Same source + same sourceHandle but different target
    const conn = { source: "a", sourceHandle: "out", target: "c", targetHandle: "in-0" };
    expect(checkValidConnection(diagram as any, conn)).toBe(true);
  });
});
