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
      connected: false,
      contextId: null,
      lastEventAt: null,
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

  it("invalidates missions and ride-details for booking_updated", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", {
      contextId: "company:42",
      missionId: 101,
    });

    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        exact: false,
      })
    );
    expect(spy.mock.calls[1]?.[0]).toEqual(
      expect.objectContaining({
        queryKey: expect.arrayContaining(["ctx", "company:42"]),
      })
    );
  });

  it("is idempotent for duplicated events received immediately", () => {
    const queryClient = new QueryClient();
    const spy = jest.spyOn(queryClient, "invalidateQueries");
    const context = { contextId: "company:42", missionId: 101 };

    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);
    invalidateCompanyQueriesForEvent(queryClient, "booking_updated", context);

    // First event does two invalidations, duplicate is ignored.
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
