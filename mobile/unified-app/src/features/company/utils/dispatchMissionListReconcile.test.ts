import { describe, expect, it } from "@jest/globals";
import type { CompanyDispatchMission } from "../api/contracts";
import {
  reconcileDispatchMissionList,
  shareDispatchMissionsQueryData,
} from "./dispatchMissionListReconcile";

function mission(
  partial: Partial<CompanyDispatchMission> & { mission_id: number }
): CompanyDispatchMission {
  return {
    status: "assigned",
    client_name: `Client ${partial.mission_id}`,
    pickup_label: "Genève",
    dropoff_label: "Lausanne",
    ...partial,
  };
}

function buildDay(count: number): CompanyDispatchMission[] {
  return Array.from({ length: count }, (_, index) => mission({ mission_id: index + 1 }));
}

describe("reconcileDispatchMissionList", () => {
  it.each([10, 30, 60, 100] as const)(
    "conserve les références des autres missions quand 1/%s change",
    (count) => {
      const previous = buildDay(count);
      const next = previous.map((item) =>
        item.mission_id === 1
          ? { ...item, driver_name: "Sonia" }
          : { ...item }
      );
      const reconciled = reconcileDispatchMissionList(previous, next);
      expect(reconciled[0]).not.toBe(previous[0]);
      expect(reconciled[0]?.driver_name).toBe("Sonia");
      for (let index = 1; index < count; index += 1) {
        expect(reconciled[index]).toBe(previous[index]);
      }
    }
  );

  it("renvoie la liste précédente si le contenu est identique", () => {
    const previous = buildDay(10);
    const next = previous.map((item) => ({ ...item }));
    expect(reconcileDispatchMissionList(previous, next)).toBe(previous);
  });
});

describe("shareDispatchMissionsQueryData", () => {
  it("ne recrée que la mission modifiée dans le payload query", () => {
    const previousMissions = buildDay(30);
    const oldData = {
      context_id: "company:1",
      refreshed_at: "t0",
      missions: previousMissions,
      total: 30,
      page_size: 50,
      loaded: 30,
      is_complete: true,
      next_page: 2,
    };
    const newData = {
      context_id: "company:1",
      refreshed_at: "t1",
      missions: previousMissions.map((item) =>
        item.mission_id === 7 ? { ...item, status: "en_route" as const } : { ...item }
      ),
      total: 30,
      page_size: 50,
      loaded: 30,
      is_complete: true,
      next_page: 2,
    };
    const shared = shareDispatchMissionsQueryData(oldData, newData);
    expect(shared.refreshed_at).toBe("t1");
    expect(shared.missions[6]).not.toBe(previousMissions[6]);
    expect(shared.missions[6]?.status).toBe("en_route");
    expect(shared.missions[5]).toBe(previousMissions[5]);
    expect(shared.missions[7]).toBe(previousMissions[7]);
  });
});
