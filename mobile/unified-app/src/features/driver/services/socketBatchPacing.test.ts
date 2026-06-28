import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";

describe("socketBatchPacing", () => {
  beforeEach(() => {
    jest.resetModules();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("blocks socket emit within min interval", () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./socketBatchPacing") as typeof import("./socketBatchPacing");
    mod.__resetSocketBatchPacingForTests();
    expect(mod.canEmitSocketBatchNow()).toBe(true);
    mod.recordSocketBatchSent();
    jest.advanceTimersByTime(2_000);
    expect(mod.canEmitSocketBatchNow()).toBe(false);
    jest.advanceTimersByTime(3_001);
    expect(mod.canEmitSocketBatchNow()).toBe(true);
  });

  it("extends cooldown after server rate limit", () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./socketBatchPacing") as typeof import("./socketBatchPacing");
    mod.__resetSocketBatchPacingForTests();
    mod.recordSocketBatchRateLimited(8_000);
    jest.advanceTimersByTime(5_000);
    expect(mod.canEmitSocketBatchNow()).toBe(false);
    jest.advanceTimersByTime(3_001);
    expect(mod.canEmitSocketBatchNow()).toBe(true);
  });
});
