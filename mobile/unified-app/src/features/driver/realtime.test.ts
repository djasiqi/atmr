import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, jest, beforeEach } from "@jest/globals";
import { driverQueryKeys } from "./queryKeys";

jest.mock("@react-native-async-storage/async-storage", () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
  },
}));

jest.mock("./api/driverHttp", () => ({
  getDriverMissions: jest.fn(),
}));

jest.mock("./services/missionSyncOrchestrator", () => ({
  scheduleDriverMissionSync: jest.fn(),
}));

const { applyDriverSocketEvent } = require("./realtime") as typeof import("./realtime");
const { missionRuntimeManager } = require("./services/missionRuntimeManager") as typeof import("./services/missionRuntimeManager");

function recentIso(offsetMs: number): string {
  return new Date(Date.now() + offsetMs).toISOString();
}

describe("driver realtime merge behavior", () => {
  beforeEach(() => {
    missionRuntimeManager.resetForTests();
  });

  it("applies event only once when sequence is duplicated", () => {
    const queryClient = new QueryClient();
    const contextId = "driver:42";
    queryClient.setQueryData(driverQueryKeys.missions(contextId), [
      {
        id: 1,
        status: "ASSIGNED",
        updated_at: recentIso(-120_000),
      },
    ]);

    applyDriverSocketEvent(queryClient, contextId, {
      mission_id: 1,
      event_type: "mission_status_changed",
      event_sequence: 2,
      updated_at: recentIso(-60_000),
      payload: { status: "EN_ROUTE" },
    });
    applyDriverSocketEvent(queryClient, contextId, {
      mission_id: 1,
      event_type: "mission_status_changed",
      event_sequence: 2,
      updated_at: recentIso(-30_000),
      payload: { status: "ARRIVED" },
    });

    const missions = queryClient.getQueryData(driverQueryKeys.missions(contextId)) as {
      id: number;
      status: string;
    }[];
    expect(missions[0].status).toBe("EN_ROUTE");
    queryClient.clear();
  });

  it("invalidates detail when ordering is stale to avoid invalidation storms", () => {
    const queryClient = new QueryClient();
    const contextId = "driver:42";
    const invalidateSpy = jest.spyOn(queryClient, "invalidateQueries");
    queryClient.setQueryData(driverQueryKeys.missions(contextId), [
      {
        id: 7,
        status: "IN_PROGRESS",
        updated_at: recentIso(0),
      },
    ]);

    applyDriverSocketEvent(queryClient, contextId, {
      mission_id: 7,
      event_type: "mission_updated",
      event_sequence: 1,
      updated_at: recentIso(-60_000),
      payload: { status: "ARRIVED" },
    });

    expect(invalidateSpy).toHaveBeenCalledTimes(1);
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: driverQueryKeys.missionDetail(contextId, 7),
    });
    queryClient.clear();
  });
});
