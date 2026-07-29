import type { DriverMission } from "../types";
import { normalizeDriverMissionStatus } from "../statusDictionary";
import { driverHasScheduledPickupTime } from "./pickupScheduling";

export type DriverDayMissionSectionKey = "todo" | "untimed" | "done";

export type DriverDayMissionSection = {
  key: DriverDayMissionSectionKey;
  label: string;
  items: DriverMission[];
};

function isCompletedMission(status: string): boolean {
  return normalizeDriverMissionStatus(status) === "COMPLETED";
}

/**
 * Sections de la liste « Courses du jour » :
 * 1. À effectuer — horaires connus (pending / accepted / assigned / en cours…)
 * 2. Heure à définir — non terminées sans horaire
 * 3. Terminées
 */
export function buildDriverDayMissionSections(
  missions: DriverMission[],
): DriverDayMissionSection[] {
  const todo: DriverMission[] = [];
  const untimed: DriverMission[] = [];
  const done: DriverMission[] = [];

  for (const mission of missions) {
    if (isCompletedMission(String(mission.status ?? ""))) {
      done.push(mission);
      continue;
    }
    if (!driverHasScheduledPickupTime(mission)) {
      untimed.push(mission);
      continue;
    }
    todo.push(mission);
  }

  return [
    { key: "todo", label: "A effectuer", items: todo },
    { key: "untimed", label: "Heure à définir", items: untimed },
    { key: "done", label: "Terminees", items: done },
  ];
}
