import { describe, expect, it, beforeEach } from "@jest/globals";
import { QueryClient } from "@tanstack/react-query";
import type { CompanyDispatchMission } from "../api/contracts";
import {
  findMissionInDispatchCache,
  peekRideDetailSnapshot,
  rememberRideDetailSnapshot,
  resetRideDetailSnapshotsForTests,
} from "./rideDetailSnapshotStore";

const mission: CompanyDispatchMission = {
  mission_id: 88,
  status: "assigned",
  client_name: "Sonia BAUER",
  driver_name: "Karim",
};

describe("rideDetailSnapshotStore", () => {
  beforeEach(() => {
    resetRideDetailSnapshotsForTests();
  });

  it("mémorise le snapshot Courses pour une ouverture immédiate", () => {
    rememberRideDetailSnapshot(mission);
    expect(peekRideDetailSnapshot(88)?.mission.client_name).toBe("Sonia BAUER");
  });

  it("ne conserve que les 2 derniers snapshots (cache secondaire borné)", () => {
    rememberRideDetailSnapshot({ ...mission, mission_id: 1 });
    rememberRideDetailSnapshot({ ...mission, mission_id: 2 });
    rememberRideDetailSnapshot({ ...mission, mission_id: 3 });
    expect(peekRideDetailSnapshot(1)).toBeNull();
    expect(peekRideDetailSnapshot(2)?.missionId).toBe(2);
    expect(peekRideDetailSnapshot(3)?.missionId).toBe(3);
  });

  it("retrouve une mission dans le cache des journées", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["ctx", "company:1", "company", "dispatch", "missions", "2026-09-05"], {
      context_id: "company:1",
      missions: [mission],
      refreshed_at: "2026-09-05T10:00:00Z",
    });
    expect(findMissionInDispatchCache(queryClient, 88)?.driver_name).toBe("Karim");
    expect(findMissionInDispatchCache(queryClient, 99)).toBeNull();
  });
});
