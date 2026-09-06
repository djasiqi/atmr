import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";

jest.mock("../../core/sessionProvider", () => ({
  useSession: () => ({
    activeContext: { context_id: "company:42", context_type: "company" },
  }),
}));

jest.mock("./api/companyApi", () => ({
  getDispatchMissions: jest.fn(),
  getCompanyRideDetail: jest.fn(),
  normalizeDispatchMission: jest.fn((raw: { mission_id?: number } | null) =>
    raw && typeof raw.mission_id === "number" ? raw : null
  ),
  getDriversLocationsSnapshot: jest.fn(),
  getOptimizerStatus: jest.fn(),
  getRealtimeDashboard: jest.fn(),
}));

jest.mock("./realtime/companyRealtimeBridge", () => ({
  companyRealtimeBridge: {
    getSnapshot: jest.fn(() => ({
      status: "idle",
      transportStatus: "idle",
      dataFreshness: "idle",
      connected: false,
      contextId: null,
      lastEventAt: null,
      lastConnectedAt: null,
      lastError: null,
    })),
    subscribe: jest.fn(() => () => undefined),
  },
}));

 
 
const { invalidateCompanyQueriesForEvent, resetCompanyInvalidationDedupStateForTests } = require("./hooks");

describe("company query invalidation policy", () => {
  beforeEach(() => {
    resetCompanyInvalidationDedupStateForTests();
  });

  it("n’invalide que le dashboard pour booking_updated + missionId (patch, pas famille rides)", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", {
      contextId: "company:42",
      missionId: 101,
    });

    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        exact: true,
        queryKey: expect.arrayContaining(["ctx", "company:42", "dashboard"]),
      })
    );
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("ride-details"))).toBe(false);
  });

  it("is idempotent for duplicated events received immediately", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");
    const context = { contextId: "company:42", missionId: 101 };

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);
    const firstPassCalls = spy.mock.calls.length;
    expect(firstPassCalls).toBe(1);
    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);

    // Doublon immédiat ignoré (dedup).
    expect(spy).toHaveBeenCalledTimes(firstPassCalls);
  });

  // Phase 2 PR B/C — gate D3.1
  it("n’invalide que le dashboard pour dispatch_assignment (J observé refetch, pas famille)", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_assignment", {
      contextId: "company:42",
    });

    expect(spy).toHaveBeenCalledTimes(1);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(false);
  });

  it("n’invalide pas ride-details ni la famille missions pour dispatch_assignment + missionId", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_assignment", {
      contextId: "company:42",
      missionId: 777,
    });

    expect(spy).toHaveBeenCalledTimes(1);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("ride-details"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(false);
  });

  it("n'invalide pas l'optimizer tant que le LOCK est OFF", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "optimizer_status_changed", {
      contextId: "company:42",
    });

    expect(spy).not.toHaveBeenCalled();
  });

  it("invalide dashboard + delays pour dispatch_run_lifecycle, pas la famille missions", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_run_lifecycle", {
      contextId: "company:42",
    });

    expect(spy).toHaveBeenCalledTimes(2);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(false);
    expect(keys.some((k) => (k as unknown[]).includes("dispatch-delays"))).toBe(true);
  });
});
