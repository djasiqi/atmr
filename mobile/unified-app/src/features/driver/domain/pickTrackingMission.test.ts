import { describe, expect, it } from "@jest/globals";
import { pickTrackingMission } from "./pickTrackingMission";
import type { DriverMission } from "../types";

function mission(
  id: number,
  status: string,
  scheduled_time: string
): DriverMission {
  return { id, status, scheduled_time };
}

describe("pickTrackingMission", () => {
  const now = Date.parse("2026-06-22T12:00:00.000Z");

  it("A EN_ROUTE + B ASSIGNED T-20 → A gagne", () => {
    const missions = [
      mission(1, "EN_ROUTE", "2026-06-22T11:30:00.000Z"),
      mission(2, "ASSIGNED", "2026-06-22T12:20:00.000Z"),
    ];
    expect(pickTrackingMission(missions, now)?.id).toBe(1);
  });

  it("A IN_PROGRESS + B ASSIGNED T-20 → A gagne", () => {
    const missions = [
      mission(1, "IN_PROGRESS", "2026-06-22T11:00:00.000Z"),
      mission(2, "ASSIGNED", "2026-06-22T12:20:00.000Z"),
    ];
    expect(pickTrackingMission(missions, now)?.id).toBe(1);
  });

  it("ignore les missions terminées", () => {
    const missions = [
      mission(1, "COMPLETED", "2026-06-22T10:00:00.000Z"),
      mission(2, "ASSIGNED", "2026-06-22T15:00:00.000Z"),
    ];
    expect(pickTrackingMission(missions, now)?.id).toBe(2);
  });
});
