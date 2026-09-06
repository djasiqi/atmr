import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";

import {
  performCompanyRecoveryResync,
  resolveRecoveryTrigger,
} from "./useCompanyRecoveryListener";
import {
  getRealtimeMetricsSnapshot,
  resetRealtimeMetricsForTests,
} from "../../../core/observability/realtimeMetrics";
import { dispatchMissionsQueryKey } from "../utils/prefetchAdjacentDispatchMissions";
import type { CompanyDispatchMissionListResponse } from "../api/contracts";

describe("recovery trigger resolver (gate D3.2)", () => {
  it("maps stale event_type", () => {
    expect(resolveRecoveryTrigger("company_data_stale_resync")).toBe("stale");
  });

  it("maps reconnect event_type", () => {
    expect(resolveRecoveryTrigger("company_socket_reconnected")).toBe("reconnect");
  });

  it("ignores unrelated events", () => {
    expect(resolveRecoveryTrigger("booking_updated")).toBeNull();
    expect(resolveRecoveryTrigger(undefined)).toBeNull();
  });
});

describe("performCompanyRecoveryResync (gate D3.2)", () => {
  beforeEach(() => {
    resetRealtimeMetricsForTests();
  });

  it("n’invalide plus la famille missions — J±1 survivent", () => {
    const queryClient = new QueryClient();
    const jKey = dispatchMissionsQueryKey("company:42", "2026-09-06");
    const prevKey = dispatchMissionsQueryKey("company:42", "2026-09-05");
    const nextKey = dispatchMissionsQueryKey("company:42", "2026-09-07");
    const neighbor: CompanyDispatchMissionListResponse = {
      context_id: "company:42",
      date: "2026-09-05",
      missions: [{ mission_id: 2, status: "pending" }],
      refreshed_at: "2026-09-06T00:00:00.000Z",
      total: 1,
      page_size: 50,
      loaded: 1,
      is_complete: true,
      next_page: 2,
      pagination_error: false,
    };
    queryClient.setQueryData(jKey, { ...neighbor, date: "2026-09-06", missions: [{ mission_id: 1, status: "pending" }] });
    queryClient.setQueryData(prevKey, neighbor);
    queryClient.setQueryData(nextKey, { ...neighbor, date: "2026-09-07", missions: [{ mission_id: 3, status: "pending" }] });

    const spy = jest.spyOn(queryClient, "invalidateQueries");
    performCompanyRecoveryResync(queryClient, "company:42", "stale");

    expect(spy).toHaveBeenCalledTimes(3);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("inbox"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("dispatch-delays"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("chat"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("institution-offers"))).toBe(false);
    expect(
      keys.some((k) => {
        const arr = k as unknown[];
        return arr.includes("drivers") && arr.includes("locations");
      })
    ).toBe(true);
    expect(queryClient.getQueryState(prevKey)?.isInvalidated).toBeFalsy();
    expect(queryClient.getQueryState(nextKey)?.isInvalidated).toBeFalsy();
    expect(queryClient.getQueryData(prevKey)).toEqual(neighbor);
  });

  it("increments recovery_resync metric by trigger", () => {
    const queryClient = new QueryClient();

    performCompanyRecoveryResync(queryClient, "company:42", "stale");
    performCompanyRecoveryResync(queryClient, "company:42", "reconnect");
    performCompanyRecoveryResync(queryClient, "company:42", "reconnect");

    const snap = getRealtimeMetricsSnapshot();
    expect(snap.recoveryResyncTotal).toBe(3);
    expect(snap.recoveryResyncByTrigger.stale).toBe(1);
    expect(snap.recoveryResyncByTrigger.reconnect).toBe(2);
  });

  it("scopes invalidations under contextScopedKey (ctx prefix)", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    performCompanyRecoveryResync(queryClient, "company:99", "reconnect");

    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.every((k) => (k as unknown[])[0] === "ctx")).toBe(true);
    expect(keys.every((k) => (k as unknown[])[1] === "company:99")).toBe(true);
  });
});
