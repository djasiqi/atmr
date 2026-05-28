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

  it("invalidates dashboard + missions + inbox + delays + chat for stale trigger", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    performCompanyRecoveryResync(queryClient, "company:42", "stale");

    expect(spy).toHaveBeenCalledTimes(5);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("inbox"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("dispatch-delays"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("chat"))).toBe(true);
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
