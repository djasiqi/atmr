import { buildDriverDayMissionSections } from "./driverDayMissionSections";
import type { DriverMission } from "../types";

function mission(partial: Partial<DriverMission> & { id: number }): DriverMission {
  return {
    status: "ASSIGNED",
    ...partial,
  } as DriverMission;
}

describe("buildDriverDayMissionSections", () => {
  it("ordonne : à effectuer (horaires connus) → heure à définir → terminées", () => {
    const timedAssigned = mission({
      id: 1,
      status: "ASSIGNED",
      scheduled_time: "2026-07-29T08:30:00+02:00",
      scheduling: { time_scheduled: true, time_defined: true },
    });
    const timedInProgress = mission({
      id: 2,
      status: "IN_PROGRESS",
      scheduled_time: "2026-07-29T09:00:00+02:00",
      scheduling: { time_scheduled: true, time_defined: true },
    });
    const done = mission({
      id: 3,
      status: "COMPLETED",
      scheduled_time: "2026-07-29T07:00:00+02:00",
      scheduling: { time_scheduled: true, time_defined: true },
    });
    const untimed = mission({
      id: 4,
      status: "PENDING",
      scheduled_time: "2026-07-29T00:00:00+02:00",
      scheduling: { time_scheduled: false, time_defined: false },
    });

    const sections = buildDriverDayMissionSections([
      untimed,
      done,
      timedInProgress,
      timedAssigned,
    ]);

    expect(sections.map((s) => s.key)).toEqual(["todo", "untimed", "done"]);
    expect(sections[0].items.map((m) => m.id)).toEqual([2, 1]);
    expect(sections[1].items.map((m) => m.id)).toEqual([4]);
    expect(sections[2].items.map((m) => m.id)).toEqual([3]);
  });

  it("met une course terminée sans horaire dans Terminées", () => {
    const doneUntimed = mission({
      id: 9,
      status: "COMPLETED",
      scheduled_time: "2026-07-29T00:00:00+02:00",
      scheduling: { time_scheduled: false, time_defined: false },
    });
    const sections = buildDriverDayMissionSections([doneUntimed]);
    expect(sections.find((s) => s.key === "done")?.items.map((m) => m.id)).toEqual([9]);
    expect(sections.find((s) => s.key === "untimed")?.items).toEqual([]);
  });
});
