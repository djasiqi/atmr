import {
  buildGenevaScheduleFromLocalCalendarDay,
  dateFromZurichWallParts,
  formatNaiveIsoInZurich,
  getTodayIsoDateInZurich,
  isoDateInZurichFromIso,
  mergeZurichDayAndTime,
  missionBelongsToSelectedDay,
  parseScheduledTimeInstant,
  scheduledTimeToFormNaiveIso,
} from "./companyDateUtils";

describe("companyDateUtils", () => {
  it("convertit une ISO UTC vers la date Zurich", () => {
    expect(isoDateInZurichFromIso("2026-06-21T22:30:00.000Z")).toBe("2026-06-22");
    expect(isoDateInZurichFromIso("2026-06-21T10:00:00+02:00")).toBe("2026-06-21");
  });

  it("exclut une mission datée sur un autre jour", () => {
    expect(
      missionBelongsToSelectedDay(
        { scheduled_at: "2026-06-20T14:00:00+02:00", time_confirmed: true },
        "2026-06-21"
      )
    ).toBe(false);
    expect(
      missionBelongsToSelectedDay(
        { scheduled_at: "2026-06-21T14:00:00+02:00", time_confirmed: true },
        "2026-06-21"
      )
    ).toBe(true);
  });

  it("conserve les missions sans horaire (retour lié)", () => {
    expect(missionBelongsToSelectedDay({ scheduled_at: null }, "2026-06-21")).toBe(true);
  });

  it("filtre les sentinelles 00:00:00 sur un autre jour", () => {
    expect(
      missionBelongsToSelectedDay(
        {
          scheduled_at: "2026-06-19T00:00:00+02:00",
          scheduling: { time_defined: false },
        },
        "2026-06-21"
      )
    ).toBe(false);
  });

  it("retourne une date du jour en timezone Zurich", () => {
    expect(getTodayIsoDateInZurich()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("parse une ISO UTC Z comme instant Genève (aligné liste courses)", () => {
    const instant = parseScheduledTimeInstant("2026-06-22T07:45:00Z");
    expect(instant).not.toBeNull();
    expect(formatNaiveIsoInZurich(instant!)).toBe("2026-06-22T09:45:00");
    expect(scheduledTimeToFormNaiveIso("2026-06-22T07:45:00Z")).toBe("2026-06-22T09:45:00");
  });

  it("conserve une ISO naïf Genève sans décalage", () => {
    expect(scheduledTimeToFormNaiveIso("2026-06-22T09:45:00")).toBe("2026-06-22T09:45:00");
    const instant = parseScheduledTimeInstant("2026-06-22T09:45:00");
    expect(formatNaiveIsoInZurich(instant!)).toBe("2026-06-22T09:45:00");
  });

  it("fusionne jour et heure en calendrier Genève", () => {
    const current = parseScheduledTimeInstant("2026-06-21T23:42:00")!;
    const tomorrow = dateFromZurichWallParts(2026, 6, 22, 0, 0, 0);
    expect(formatNaiveIsoInZurich(mergeZurichDayAndTime(tomorrow, current))).toBe(
      "2026-06-22T23:42:00",
    );
    const localStripDay = new Date(2026, 5, 22);
    expect(formatNaiveIsoInZurich(buildGenevaScheduleFromLocalCalendarDay(localStripDay, current))).toBe(
      "2026-06-22T23:42:00",
    );
  });
});
