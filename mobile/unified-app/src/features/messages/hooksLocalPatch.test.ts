import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import {
  buildCompanyHubCacheKeys,
  buildDriverHubCacheKeys,
  getThreadUnreadCount,
  patchThreadsOnRead,
  patchUnreadOnRead,
} from "./messageHubCachePatch";

describe("hub local patch flows (no invalidate)", () => {
  let qc: QueryClient;

  beforeEach(() => {
    qc = new QueryClient();
  });

  it("driver mark-read patch does not invalidate queries", async () => {
    const keys = buildDriverHubCacheKeys(7, 100);
    qc.setQueryData(keys.threads, {
      threads: [{ thread_id: "dispatch", section: "dispatch", title: "D", unread_count: 2 }],
      unread_total: 2,
    });
    qc.setQueryData(keys.unread, 2);
    const invalidateSpy = jest.spyOn(qc, "invalidateQueries");

    const delta = getThreadUnreadCount(qc, keys, "dispatch");
    await patchThreadsOnRead(qc, keys, "dispatch");
    await patchUnreadOnRead(qc, keys, delta);

    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(qc.getQueryData(keys.unread)).toBe(0);
  });

  it("company mark-read patch does not invalidate queries", async () => {
    const keys = buildCompanyHubCacheKeys(7);
    qc.setQueryData(keys.threads, {
      threads: [{ thread_id: "team", section: "team", title: "T", unread_count: 1 }],
      unread_total: 1,
    });
    qc.setQueryData(keys.unread, 1);
    const invalidateSpy = jest.spyOn(qc, "invalidateQueries");

    const delta = getThreadUnreadCount(qc, keys, "team");
    await patchThreadsOnRead(qc, keys, "team");
    await patchUnreadOnRead(qc, keys, delta);

    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(qc.getQueryData(keys.unread)).toBe(0);
  });
});
