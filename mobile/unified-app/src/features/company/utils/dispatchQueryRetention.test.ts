import { beforeEach, describe, expect, it } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import { dispatchMissionsQueryKey } from "./prefetchAdjacentDispatchMissions";
import {
  datesToRetain,
  pruneDispatchQueryCache,
  rememberVisitedDispatchDate,
  resetDispatchQueryRetentionForTests,
} from "./dispatchQueryRetention";

function seedDay(queryClient: QueryClient, date: string): unknown[] {
  const key = dispatchMissionsQueryKey("company:42", date);
  queryClient.setQueryData(key, {
    context_id: "company:42",
    date,
    missions: [{ mission_id: Number(date.slice(-2)), status: "pending" }],
    total: 1,
    loaded: 1,
    page_size: 50,
    is_complete: true,
    next_page: 2,
    refreshed_at: "2026-09-06T00:00:00.000Z",
  });
  return key;
}

describe("dispatchQueryRetention (OPT-06)", () => {
  beforeEach(() => {
    resetDispatchQueryRetentionForTests();
  });

  it("pin J + J±1 et n’ajoute que 2 extras LRU", () => {
    rememberVisitedDispatchDate("2026-09-06");
    rememberVisitedDispatchDate("2026-09-07");
    rememberVisitedDispatchDate("2026-09-08");
    rememberVisitedDispatchDate("2026-09-09");
    rememberVisitedDispatchDate("2026-09-10");
    const keep = datesToRetain(["2026-09-10"]);
    expect(keep.has("2026-09-09")).toBe(true);
    expect(keep.has("2026-09-10")).toBe(true);
    expect(keep.has("2026-09-11")).toBe(true);
    expect(keep.has("2026-09-08")).toBe(true);
    expect(keep.has("2026-09-07")).toBe(true);
    expect(keep.has("2026-09-06")).toBe(false);
  });

  it("évacue les journées hors fenêtre sans toucher J±1", () => {
    const queryClient = new QueryClient();
    for (const date of ["2026-09-06", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"]) {
      rememberVisitedDispatchDate(date);
      seedDay(queryClient, date);
    }
    const result = pruneDispatchQueryCache(queryClient, "company:42", ["2026-09-10"]);
    expect(queryClient.getQueryData(dispatchMissionsQueryKey("company:42", "2026-09-10"))).toBeTruthy();
    expect(queryClient.getQueryData(dispatchMissionsQueryKey("company:42", "2026-09-09"))).toBeTruthy();
    expect(queryClient.getQueryData(dispatchMissionsQueryKey("company:42", "2026-09-06"))).toBeUndefined();
    expect(result.removed_dates).toContain("2026-09-06");
    expect(result.mission_days).toBeGreaterThanOrEqual(3);
  });
});
