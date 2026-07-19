import { describe, expect, it } from "vitest";
import { canCancelTrainingJob } from "../training/api";

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
