import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import {
  buildRefreshScopeKey,
  flushAllScheduledRefreshesForTests,
  resetNotificationRefreshSchedulerForTests,
  scheduleScopedRefresh,
} from "./notificationRefreshScheduler";

describe("notificationRefreshScheduler", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    resetNotificationRefreshSchedulerForTests();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("coalesces refresh per scope", () => {
    const flush = jest.fn();
    const scope = { kind: "missionDetail" as const, contextId: "ctx-1", missionId: 42 };
    scheduleScopedRefresh(scope, flush);
    scheduleScopedRefresh(scope, flush);
    expect(flush).not.toHaveBeenCalled();
    jest.advanceTimersByTime(800);
    expect(flush).toHaveBeenCalledTimes(1);
  });

  it("keeps distinct scopes separate", () => {
    const flushA = jest.fn();
    const flushB = jest.fn();
    scheduleScopedRefresh({ kind: "missions", contextId: "ctx-1" }, flushA);
    scheduleScopedRefresh({ kind: "chat", contextId: "ctx-1", threadId: "t-1" }, flushB);
    jest.advanceTimersByTime(800);
    expect(flushA).toHaveBeenCalledTimes(1);
    expect(flushB).toHaveBeenCalledTimes(1);
    expect(buildRefreshScopeKey({ kind: "missions", contextId: "ctx-1" })).not.toBe(
      buildRefreshScopeKey({ kind: "chat", contextId: "ctx-1", threadId: "t-1" })
    );
  });

  it("flush helper drains pending scopes in tests", () => {
    const flush = jest.fn();
    scheduleScopedRefresh({ kind: "missions", contextId: "ctx-2" }, flush);
    flushAllScheduledRefreshesForTests();
    expect(flush).toHaveBeenCalledTimes(1);
  });
});
