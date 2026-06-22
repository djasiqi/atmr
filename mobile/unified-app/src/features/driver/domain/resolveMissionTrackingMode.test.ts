import { describe, expect, it } from "@jest/globals";
import {
  isOperationalDepartureWithinLeadMinutes,
  resolveMissionTrackingMode,
} from "./resolveMissionTrackingMode";
import type { DriverMission } from "../types";

function mission(partial: Partial<DriverMission> & { status: string }): DriverMission {
  return {
    id: 1,
    status: partial.status,
    scheduled_time: partial.scheduled_time ?? null,
    time_confirmed: partial.time_confirmed ?? null,
    scheduling: partial.scheduling ?? null,
  };
}

describe("resolveMissionTrackingMode", () => {
  const now = Date.parse("2026-06-22T12:00:00.000Z");

  it("returns mission_live for EN_ROUTE, ARRIVED, IN_PROGRESS", () => {
    expect(resolveMissionTrackingMode(mission({ status: "EN_ROUTE" }), now)).toBe("mission_live");
    expect(resolveMissionTrackingMode(mission({ status: "ARRIVED" }), now)).toBe("mission_live");
    expect(resolveMissionTrackingMode(mission({ status: "IN_PROGRESS" }), now)).toBe(
      "mission_live"
    );
  });

  it("ASSIGNED sans heure planifiée → availability_presence", () => {
    expect(
      resolveMissionTrackingMode(
        mission({ status: "ASSIGNED", scheduled_time: null, time_confirmed: true }),
        now
      )
    ).toBe("availability_presence");
  });

  it("ASSIGNED legacy T00:00:00 → availability_presence", () => {
    expect(
      resolveMissionTrackingMode(
        mission({ status: "ASSIGNED", scheduled_time: "2026-06-22T00:00:00.000Z" }),
        now
      )
    ).toBe("availability_presence");
  });

  it("ASSIGNED RDV dans 2 h → availability_presence", () => {
    expect(
      resolveMissionTrackingMode(
        mission({ status: "ASSIGNED", scheduled_time: "2026-06-22T14:00:00.000Z" }),
        now
      )
    ).toBe("availability_presence");
  });

  it("ASSIGNED RDV dans 20 min → mission_live", () => {
    expect(
      resolveMissionTrackingMode(
        mission({ status: "ASSIGNED", scheduled_time: "2026-06-22T12:20:00.000Z" }),
        now
      )
    ).toBe("mission_live");
  });

  it("statuts terminaux → null", () => {
    for (const status of ["COMPLETED", "CANCELLED", "NO_SHOW", "EXPIRED"] as const) {
      expect(resolveMissionTrackingMode(mission({ status }), now)).toBeNull();
    }
  });

  it("isOperationalDepartureWithinLeadMinutes respecte la fenêtre T-30", () => {
    const m = mission({ status: "ASSIGNED", scheduled_time: "2026-06-22T12:29:00.000Z" });
    expect(isOperationalDepartureWithinLeadMinutes(m, now, 30)).toBe(true);
    expect(
      isOperationalDepartureWithinLeadMinutes(
        mission({ status: "ASSIGNED", scheduled_time: "2026-06-22T12:31:00.000Z" }),
        now,
        30
      )
    ).toBe(false);
  });
});
