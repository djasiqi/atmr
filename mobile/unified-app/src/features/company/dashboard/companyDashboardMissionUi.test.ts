import {
  isMissionDelayed,
  missionHasConfirmedPickupTime,
  missionHasDefinedPickupTime,
  resolveMissionUiStatus,
} from "./companyDashboardMissionUi";

describe("companyDashboardMissionUi", () => {
  it("detects pickup sentinel as unconfirmed schedule", () => {
    expect(missionHasDefinedPickupTime("2026-05-18T00:00:00+02:00")).toBe(false);
    expect(missionHasDefinedPickupTime("2026-05-18T08:00:00+02:00")).toBe(true);
  });

  it("does not mark TBD pickup time missions as delayed", () => {
    const nowMs = Date.parse("2026-05-18T15:00:00+02:00");
    const status = resolveMissionUiStatus(
      {
        mission_id: 1,
        status: "assigned",
        scheduled_at: "2026-05-18T00:00:00+02:00",
        assignment_pickup_delay_minutes: 12,
      },
      nowMs
    );
    expect(status.tone).not.toBe("delayed");
    expect(status.label).toBe("Heure à définir");
    expect(isMissionDelayed(
      {
        mission_id: 1,
        status: "assigned",
        scheduled_at: "2026-05-18T00:00:00+02:00",
        assignment_pickup_delay_minutes: 12,
      },
      nowMs
    )).toBe(false);
  });

  it("still marks past confirmed scheduled missions as delayed", () => {
    const nowMs = Date.parse("2026-05-18T15:00:00+02:00");
    const status = resolveMissionUiStatus(
      {
        mission_id: 2,
        status: "assigned",
        scheduled_at: "2026-05-18T08:00:00+02:00",
        time_confirmed: true,
      },
      nowMs
    );
    expect(status.tone).toBe("delayed");
    expect(status.label).toBe("En retard");
    expect(
      missionHasConfirmedPickupTime({
        mission_id: 2,
        status: "assigned",
        scheduled_at: "2026-05-18T08:00:00+02:00",
        time_confirmed: true,
      })
    ).toBe(true);
  });
});
