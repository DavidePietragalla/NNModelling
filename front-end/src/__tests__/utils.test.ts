/*
 * NNModelling — DSL for designing neural networks via visual node editor
 * Copyright (C) 2026  Luca Sforza
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import { describe, it, expect, afterAll } from "vitest";
import { type Edge } from "@xyflow/svelte";
import { Diagram } from "../Diagram.svelte";
import { checkValidConnection } from "../utils";
import { stubWindow, unstubWindow } from "./helpers";

stubWindow();
afterAll(() => unstubWindow());

describe("checkValidConnection", () => {
  it("allows connection when no edges exist", () => {
    const d = new Diagram();
    d.edges = [];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(d, conn)).toBe(true);
  });

  it("allows connection when target is different from all existing edges", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "y", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(d, conn)).toBe(true);
  });

  it("allows connection when same target but different handle", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "b", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-1" };
    expect(checkValidConnection(d, conn)).toBe(true);
  });

  it("blocks connection when same target + same targetHandle is taken", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "b", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: "in-0" };
    expect(checkValidConnection(d, conn)).toBe(false);
  });

  it("blocks when passed an existing Edge instead of Connection", () => {
    const existingEdge: Edge = {
      id: "e1",
      source: "x",
      target: "b",
      targetHandle: "in-0",
    };
    const d = new Diagram();
    d.edges = [existingEdge];
    expect(checkValidConnection(d, existingEdge)).toBe(false);
  });

  it("allows connection when target handle is null and free", () => {
    const d = new Diagram();
    d.edges = [];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: null };
    expect(checkValidConnection(d, conn)).toBe(true);
  });

  it("blocks connection when target handle is null but another edge with null already exists", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "b" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "b", targetHandle: null };
    expect(checkValidConnection(d, conn)).toBe(false);
  });

  it("allows connection to join node with multiple inputs when specific free handle exists", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "join1", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "join1", targetHandle: "in-1" };
    expect(checkValidConnection(d, conn)).toBe(true);
  });

  it("blocks connection to join node when specific handle is already used", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "x", target: "join1", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: null, target: "join1", targetHandle: "in-0" };
    expect(checkValidConnection(d, conn)).toBe(false);
  });

  it("allows source handle duplicates (source handles not checked)", () => {
    const d = new Diagram();
    d.edges = [
      { id: "e1", source: "a", sourceHandle: "out", target: "b", targetHandle: "in-0" } as Edge,
    ];
    const conn = { source: "a", sourceHandle: "out", target: "c", targetHandle: "in-0" };
    expect(checkValidConnection(d, conn)).toBe(true);
  });
});
