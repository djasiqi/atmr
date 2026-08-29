import { beforeEach, describe, expect, it, jest } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import AsyncStorage from "@react-native-async-storage/async-storage";

const callOrder: string[] = [];
const mockGetDriverMissionsSince = jest.fn<(...args: any[]) => any>();
const mockGetDriverMissions = jest.fn<(...args: any[]) => any>();
const mockFlush = jest.fn<(...args: any[]) => any>();

jest.mock("./api", () => ({
  getDriverMissionsSince: (...args: unknown[]) => {
    callOrder.push("fetch");
    return mockGetDriverMissionsSince(...args);
  },
  getDriverMissions: (...args: unknown[]) => {
    callOrder.push("fetch");
    return mockGetDriverMissions(...args);
  },
}));

jest.mock("./offlineQueue", () => ({
  driverOfflineQueue: {
    flush: (...args: unknown[]) => {
      callOrder.push("flush");
      return mockFlush(...args);
    },
  },
}));

jest.mock("../../core/observability/driverTelemetry", () => ({
  emitDriverTelemetry: jest.fn(),
}));

jest.mock("../../core/featureFlags/registry", () => ({
  isFeatureEnabled: jest.fn(() => false),
}));

jest.mock("../../core/realtime/realtimeManager", () => ({
  realtimeManager: { setTransportAuthority: jest.fn() },
}));

jest.mock("./services/missionState", () => ({
  setActiveMissionFromList: jest.fn(async () => undefined),
}));

jest.mock("./services/missionRuntimeManager", () => ({
  missionRuntimeManager: { registerSnapshot: jest.fn() },
}));

import { reconcileDriverMissions } from "./sync";
import { driverQueryKeys } from "./queryKeys";
import { resetDriverMilestoneOverlayForTests } from "./domain/missionMilestoneOverlay";
import type { DriverMission } from "./types";

const CONTEXT_ID = "driver:99";

describe("reconcileDriverMissions (M1 + M2 + ordre flush→fetch)", () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    callOrder.length = 0;
    resetDriverMilestoneOverlayForTests();
    await AsyncStorage.clear();
    mockFlush.mockResolvedValue({ sent: 0, dropped: 0, failed: 0 });
  });

  it("rejoue l'outbox AVANT de lire l'état serveur", async () => {
    mockGetDriverMissionsSince.mockResolvedValue([]);
    const queryClient = new QueryClient();
    await reconcileDriverMissions(queryClient, CONTEXT_ID);
    expect(callOrder).toEqual(["flush", "fetch"]);
  });

  it("M1 : le delta passe par mapDriverMission (composition ARRIVED conservée)", async () => {
    mockGetDriverMissionsSince.mockResolvedValue([
      {
        id: 2,
        status: "en_route",
        mission_milestone: "ARRIVED",
        updated_at: "2026-08-27T10:00:00Z",
        mission_revision: 3,
        assignment_id: 20,
      },
    ]);
    const queryClient = new QueryClient();
    await reconcileDriverMissions(queryClient, CONTEXT_ID);
    const missions =
      (queryClient.getQueryData(driverQueryKeys.missions(CONTEXT_ID)) as DriverMission[]) ?? [];
    expect(missions).toHaveLength(1);
    expect(missions[0]!.status).toBe("ARRIVED");
    expect(missions[0]!.mission_revision).toBe(3);
    expect(missions[0]!.assignment_id).toBe(20);
  });

  it("M2 : un delta périmé (revision inférieure) n'écrase pas l'état local", async () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(driverQueryKeys.missions(CONTEXT_ID), [
      {
        id: 5,
        status: "IN_PROGRESS",
        assignment_id: 50,
        mission_revision: 6,
      } satisfies Partial<DriverMission> as DriverMission,
    ]);
    mockGetDriverMissionsSince.mockResolvedValue([
      {
        id: 5,
        status: "en_route",
        assignment_id: 50,
        mission_revision: 2,
        updated_at: "2026-08-27T09:00:00Z",
      },
    ]);
    await reconcileDriverMissions(queryClient, CONTEXT_ID);
    const missions =
      (queryClient.getQueryData(driverQueryKeys.missions(CONTEXT_ID)) as DriverMission[]) ?? [];
    expect(missions).toHaveLength(1);
    expect(missions[0]!.status).toBe("IN_PROGRESS");
    expect(missions[0]!.mission_revision).toBe(6);
  });

  it("la réconciliation continue même si le flush échoue", async () => {
    mockFlush.mockRejectedValueOnce(new Error("offline"));
    mockGetDriverMissionsSince.mockResolvedValue([]);
    const queryClient = new QueryClient();
    const result = await reconcileDriverMissions(queryClient, CONTEXT_ID);
    expect(result.queue).toEqual({ sent: 0, dropped: 0, failed: 0 });
    expect(callOrder).toEqual(["flush", "fetch"]);
  });
});
