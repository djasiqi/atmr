import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import {
  appendMessageToThreadCache,
  buildCompanyHubCacheKeys,
  markMessageFailedInCache,
  markTimedOutOptimisticMessages,
  messageSortKey,
  OPTIMISTIC_MESSAGE_TIMEOUT_MS,
  patchThreadsOnRead,
  patchThreadsOnReceive,
  patchUnreadOnRead,
  replaceMessageInThreadCache,
  safePatch,
  type PatchableHubMessage,
} from "./messageHubCachePatch";

function msg(partial: Partial<PatchableHubMessage> & { id: string | number }): PatchableHubMessage {
  return {
    sender_id: 1,
    content: "hi",
    timestamp: "2026-01-01T10:00:00.000Z",
    message_type: "text",
    priority: "normal",
    ...partial,
  };
}

describe("messageHubCachePatch", () => {
  let qc: QueryClient;
  const keys = buildCompanyHubCacheKeys(42);

  beforeEach(() => {
    qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    qc.setQueryData(keys.threads, {
      threads: [
        {
          thread_id: "dispatch",
          section: "dispatch",
          title: "Dispatch",
          unread_count: 3,
          last_message_at: "2026-01-01T09:00:00.000Z",
        },
        {
          thread_id: "team",
          section: "team",
          title: "Team",
          unread_count: 0,
          last_message_at: "2026-01-01T08:00:00.000Z",
        },
      ],
      unread_total: 3,
    });
    qc.setQueryData(keys.unread, 3);
    qc.setQueryData(keys.messages("dispatch"), [
      msg({ id: "1", content: "a", timestamp: "2026-01-01T10:00:00.000Z" }),
    ]);
  });

  it("calls cancelQueries before setQueryData in safePatch", async () => {
    const cancelSpy = jest.spyOn(qc, "cancelQueries");
    const setSpy = jest.spyOn(qc, "setQueryData");
    await safePatch<number>(qc, keys.unread, (prev) => (prev ?? 0) + 1);
    expect(cancelSpy).toHaveBeenCalledWith({ queryKey: keys.unread });
    expect(cancelSpy.mock.invocationCallOrder[0]).toBeLessThan(
      setSpy.mock.invocationCallOrder[0] ?? 0
    );
  });

  it("patchUnreadOnRead clamps at zero", async () => {
    await patchUnreadOnRead(qc, keys, 99);
    expect(qc.getQueryData(keys.unread)).toBe(0);
  });

  it("patchThreadsOnRead clears thread unread_count", async () => {
    const cleared = await patchThreadsOnRead(qc, keys, "dispatch");
    expect(cleared).toBe(3);
    const threads = qc.getQueryData<{ threads: { thread_id: string; unread_count: number }[] }>(
      keys.threads
    );
    expect(threads?.threads.find((t) => t.thread_id === "dispatch")?.unread_count).toBe(0);
  });

  it("appendMessageToThreadCache is idempotent by id and client_id", async () => {
    const m = msg({
      id: "local-1",
      _localId: "local-1",
      status: "sending",
      optimisticTimestamp: "2026-01-01T11:00:00.000Z",
    });
    await appendMessageToThreadCache(qc, keys, "dispatch", m);
    await appendMessageToThreadCache(qc, keys, "dispatch", { ...m, content: "dup" });
    const list = qc.getQueryData<PatchableHubMessage[]>(keys.messages("dispatch"));
    expect(list?.length).toBe(2);
    expect(list?.find((row) => row._localId === "local-1")?.content).toBe("dup");
  });

  it("keeps stable chronological order after optimistic, ack, realtime", async () => {
    const optimistic = msg({
      id: "local-x",
      _localId: "local-x",
      status: "sending",
      optimisticTimestamp: "2026-01-01T12:00:00.000Z",
      timestamp: "2026-01-01T12:00:00.000Z",
    });
    await appendMessageToThreadCache(qc, keys, "dispatch", optimistic);
    await replaceMessageInThreadCache(qc, keys, "dispatch", "local-x", {
      ...optimistic,
      id: 900,
      timestamp: "2026-01-01T12:00:01.000Z",
      status: "sent",
    });
    await appendMessageToThreadCache(
      qc,
      keys,
      "dispatch",
      msg({
        id: 900,
        content: "a",
        timestamp: "2026-01-01T12:00:01.000Z",
      })
    );
    const list = qc.getQueryData<PatchableHubMessage[]>(keys.messages("dispatch")) ?? [];
    expect(list.map((m) => String(m.id))).toEqual(["1", "900"]);
    expect(messageSortKey(list[1]!)).toBeGreaterThan(messageSortKey(list[0]!));
  });

  it("patchThreadsOnReceive bumps last_message_at order", async () => {
    await patchThreadsOnReceive(
      qc,
      keys,
      "dispatch",
      msg({ id: 2, content: "new", timestamp: "2026-01-02T10:00:00.000Z" }),
      null
    );
    const threads = qc.getQueryData<{ threads: { thread_id: string }[] }>(keys.threads);
    expect(threads?.threads[0]?.thread_id).toBe("dispatch");
  });

  it("markTimedOutOptimisticMessages marks sending older than TTL as failed", async () => {
    const old = Date.now() - OPTIMISTIC_MESSAGE_TIMEOUT_MS - 1000;
    qc.setQueryData(keys.messages("dispatch"), [
      msg({
        id: "local-old",
        _localId: "local-old",
        status: "sending",
        optimisticTimestamp: new Date(old).toISOString(),
        timestamp: new Date(old).toISOString(),
      }),
    ]);
    await markTimedOutOptimisticMessages(qc, keys, Date.now(), ["dispatch"]);
    const list = qc.getQueryData<PatchableHubMessage[]>(keys.messages("dispatch")) ?? [];
    expect(list[0]?.status).toBe("failed");
    expect(list[0]?.failure_reason).toBe("timeout");
  });

  it("no-op append when messages cache is absent", async () => {
    const fresh = new QueryClient();
    const had = await appendMessageToThreadCache(
      fresh,
      keys,
      "missing",
      msg({ id: "x" })
    );
    expect(had).toBe(false);
    expect(fresh.getQueryData(keys.messages("missing"))).toBeUndefined();
  });

  it("markMessageFailedInCache preserves message for retry", async () => {
    await appendMessageToThreadCache(
      qc,
      keys,
      "dispatch",
      msg({ id: "local-1", _localId: "local-1", status: "sending" })
    );
    await markMessageFailedInCache(qc, keys, "dispatch", "local-1", "timeout");
    const list = qc.getQueryData<PatchableHubMessage[]>(keys.messages("dispatch")) ?? [];
    const failed = list.find((m) => m._localId === "local-1");
    expect(failed?.status).toBe("failed");
  });
});
