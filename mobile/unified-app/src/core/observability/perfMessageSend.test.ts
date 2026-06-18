import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";

import { resetPerfInstrumentationForTests } from "./perfInstrumentation";
import { setPerfKpiSink } from "./perfKpi";
import {
  endMessageSend,
  recordChatCacheMismatch,
  recordMessageSendRetry,
  startMessageSend,
} from "./perfMessageSend";
import {
  applyOptimisticSendMutate,
  clearOptimisticSendTimer,
  scheduleOptimisticSendTimeout,
} from "../../features/messages/hubSendMutationHelpers";
import { buildCompanyHubCacheKeys } from "../../features/messages/messageHubCachePatch";

jest.mock("./perfInstrumentationTier", () => ({
  shouldRecordPerfMetric: () => true,
  shouldEmitPerfEventPerCall: () => false,
}));

jest.mock("./perfActiveContext", () => ({
  getPerfActiveContext: () => ({ role: "driver", screen: "/messages/test" }),
}));

jest.mock("../featureFlags/registry", () => ({
  isFeatureEnabled: () => true,
}));

jest.mock("../../features/messages/chatLocalPatchFlag", () => ({
  isPerfChatLocalPatchEnabled: () => true,
}));

describe("perf.message.send", () => {
  const events: { event: string; payload: Record<string, unknown> }[] = [];

  beforeEach(() => {
    jest.useFakeTimers();
    resetPerfInstrumentationForTests();
    events.length = 0;
    setPerfKpiSink((event, payload) => {
      events.push({ event, payload: payload as Record<string, unknown> });
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    setPerfKpiSink(null);
  });

  it("records timeout when optimistic send exceeds TTL", async () => {
    const qc = new QueryClient();
    const keys = buildCompanyHubCacheKeys(1);
    qc.setQueryData(keys.messages("dispatch"), []);

    const handle = startMessageSend({
      role: "company",
      threadId: "dispatch",
      clientId: "local-timeout",
    });
    const timer = scheduleOptimisticSendTimeout({
      qc,
      keys,
      threadId: "dispatch",
      clientId: "local-timeout",
      role: "company",
      perfHandle: handle,
    });
    expect(timer).not.toBeNull();
    jest.advanceTimersByTime(30_000);
    const timeoutEvents = events.filter((e) => e.payload.metric === "timeout_ms");
    expect(timeoutEvents.length).toBeGreaterThan(0);
    if (timer) clearTimeout(timer);
  });

  it("cancels timeout timer on success path cleanup", async () => {
    const qc = new QueryClient();
    const keys = buildCompanyHubCacheKeys(1);
    const ctx = await applyOptimisticSendMutate({
      qc,
      keys,
      threadId: "dispatch",
      optimistic: {
        id: "local-ok",
        _localId: "local-ok",
        sender_id: 1,
        content: "x",
        timestamp: new Date().toISOString(),
        status: "sending",
        message_type: "text",
        priority: "normal",
      },
      role: "company",
    });
    clearOptimisticSendTimer(ctx);
    jest.advanceTimersByTime(30_000);
    const list = qc.getQueryData<{ status?: string }[]>(keys.messages("dispatch")) ?? [];
    const row = list.find((m) => (m as { _localId?: string })._localId === "local-ok");
    expect(row?.status).not.toBe("failed");
  });

  it("increments retry_count once per retry", () => {
    recordMessageSendRetry({
      role: "driver",
      threadId: "dispatch",
      clientId: "local-r",
    });
    endMessageSend(
      startMessageSend({ role: "driver", threadId: "dispatch", clientId: "local-r" }),
      "optimistic"
    );
    const retryEvents = events.filter((e) => e.payload.metric === "retry_count");
    expect(retryEvents).toHaveLength(1);
  });

  it("emits unread_drift via recordChatCacheMismatch", () => {
    recordChatCacheMismatch({
      kind: "unread_drift",
      role: "driver",
      details: { local: 5, server: 0, delta: 5 },
    });
    expect(events.some((e) => e.event === "perf.chat_cache_mismatch")).toBe(true);
  });
});
