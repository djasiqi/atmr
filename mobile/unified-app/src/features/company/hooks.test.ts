import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";

jest.mock("../../core/sessionProvider", () => ({
  useSession: () => ({
    activeContext: { context_id: "company:42", context_type: "company" },
  }),
}));

jest.mock("./api/companyApi", () => ({
  getDispatchMissions: jest.fn(),
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

// eslint-disable-next-line @typescript-eslint/no-require-imports
const {
  invalidateCompanyQueriesForEvent,
  resetCompanyInvalidationDedupStateForTests,
} = require("./hooks");

describe("company query invalidation policy", () => {
  beforeEach(() => {
    resetCompanyInvalidationDedupStateForTests();
  });

  it("invalidates ride-details and dashboard for booking_updated with missionId", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", {
      contextId: "company:42",
      missionId: 101,
    });

    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy.mock.calls.every((call) => (call[0] as { exact?: boolean })?.exact === true)).toBe(
      true
    );
    expect(spy.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        queryKey: expect.arrayContaining(["ctx", "company:42", "ride-details"]),
      })
    );
    expect(spy.mock.calls[1]?.[0]).toEqual(
      expect.objectContaining({
        queryKey: expect.arrayContaining(["ctx", "company:42", "dashboard"]),
      })
    );
  });

  it("is idempotent for duplicated events received immediately", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");
    const context = { contextId: "company:42", missionId: 101 };

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);
    const firstPassCalls = spy.mock.calls.length;
    expect(firstPassCalls).toBe(2);
    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);

    // Doublon immédiat ignoré (dedup).
    expect(spy).toHaveBeenCalledTimes(firstPassCalls);
  });

  // Phase 2 PR B/C — gate D3.1
  it("invalidates dashboard + missions for dispatch_assignment without missionId", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_assignment", {
      contextId: "company:42",
    });

    expect(spy).toHaveBeenCalledTimes(2);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(true);
  });

  it("invalidates dashboard + missions + ride-details for dispatch_assignment with missionId", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_assignment", {
      contextId: "company:42",
      missionId: 777,
    });

    expect(spy).toHaveBeenCalledTimes(3);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("ride-details"))).toBe(true);
  });

  it("invalidates dashboard + missions + dispatch-delays for dispatch_run_lifecycle", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "dispatch_run_lifecycle", {
      contextId: "company:42",
    });

    expect(spy).toHaveBeenCalledTimes(3);
    const keys = spy.mock.calls.map((call) => (call[0] as { queryKey: unknown[] }).queryKey);
    expect(keys.some((k) => (k as unknown[]).includes("dashboard"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("missions"))).toBe(true);
    expect(keys.some((k) => (k as unknown[]).includes("dispatch-delays"))).toBe(true);
  });
});
