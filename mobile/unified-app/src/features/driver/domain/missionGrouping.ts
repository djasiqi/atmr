import type { DriverMission } from "../types";

const GROUPING_WINDOW_MS = 5 * 60 * 1000;

function normalizeAddress(value: unknown): string {
  if (typeof value !== "string") return "";
  return value
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[.,]/g, "")
    .trim()
    .slice(0, 80);
}

function getScheduledEpoch(mission: DriverMission): number {
  const raw = mission.scheduled_time;
  if (typeof raw !== "string" || raw.length === 0) return Number.MAX_SAFE_INTEGER;
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER;
}

function isTerminalStatus(mission: DriverMission): boolean {
  const status = String(mission.status ?? "").toLowerCase();
  return status === "completed" || status === "cancelled" || status === "canceled" || status === "failed";
}

export type DriverMissionGroup = {
  id: string;
  displayLabel: string;
  missions: DriverMission[];
  isGrouped: boolean;
};

export function filterNextMissionsOnly(missions: DriverMission[]): DriverMission[] {
  if (!Array.isArray(missions) || missions.length === 0) return [];

  const active = missions.filter((mission) => {
    const status = String(mission.status ?? "").toLowerCase();
    return status === "in_progress" || status === "en_route";
  });

  // Keep all missions sharing pickup + 5min window with currently active mission(s).
  if (active.length > 0) {
    const grouped = missions.filter((mission) => {
      if (isTerminalStatus(mission)) return false;
      const missionPickup = normalizeAddress(mission.pickup_location);
      const missionEpoch = getScheduledEpoch(mission);
      return active.some((entry) => {
        const pickup = normalizeAddress(entry.pickup_location);
        if (!pickup || pickup !== missionPickup) return false;
        const diff = Math.abs(getScheduledEpoch(entry) - missionEpoch);
        return diff <= GROUPING_WINDOW_MS;
      });
    });
    if (grouped.length > 0) return grouped;
  }

  const assigned = missions
    .filter((mission) => !isTerminalStatus(mission))
    .sort((a, b) => getScheduledEpoch(a) - getScheduledEpoch(b));

  const first = assigned[0];
  if (!first) return [];

  const firstPickup = normalizeAddress(first.pickup_location);
  const firstEpoch = getScheduledEpoch(first);

  return assigned.filter((mission) => {
    if (!firstPickup) return mission.id === first.id;
    const pickup = normalizeAddress(mission.pickup_location);
    if (pickup !== firstPickup) return false;
    const diff = Math.abs(getScheduledEpoch(mission) - firstEpoch);
    return diff <= GROUPING_WINDOW_MS;
  });
}

export function groupMissionsByPickupWindow(missions: DriverMission[]): DriverMissionGroup[] {
  if (!Array.isArray(missions) || missions.length === 0) return [];

  const sorted = [...missions].sort((a, b) => getScheduledEpoch(a) - getScheduledEpoch(b));
  const groups: DriverMissionGroup[] = [];

  sorted.forEach((mission) => {
    const pickup = normalizeAddress(mission.pickup_location);
    const scheduledEpoch = getScheduledEpoch(mission);
    if (!pickup) {
      groups.push({
        id: `single-${mission.id}`,
        displayLabel: String(mission.pickup_location ?? "Depart non renseigne"),
        missions: [mission],
        isGrouped: false,
      });
      return;
    }

    const existing = groups.find((group) => {
      const head = group.missions[group.missions.length - 1];
      const headPickup = normalizeAddress(head.pickup_location);
      if (headPickup !== pickup) return false;
      const diff = Math.abs(getScheduledEpoch(head) - scheduledEpoch);
      return diff <= GROUPING_WINDOW_MS;
    });

    if (!existing) {
      groups.push({
        id: `pickup-${pickup}-${scheduledEpoch}`,
        displayLabel: String(mission.pickup_location ?? "Depart non renseigne"),
        missions: [mission],
        isGrouped: false,
      });
      return;
    }

    existing.missions.push(mission);
    existing.isGrouped = existing.missions.length > 1;
  });

  return groups;
}

