import type { DriverMission, DriverMissionStatus } from "../types";
import { driverHasScheduledPickupTime } from "../utils/pickupScheduling";

export type MissionTrackingMode = "mission_live" | "availability_presence";

export const MISSION_TRACKING_LEAD_MINUTES = Number(
  process.env.EXPO_PUBLIC_MISSION_TRACKING_LEAD_MINUTES ?? "30"
);

export const TRACKING_TERMINAL_STATUSES: DriverMissionStatus[] = [
  "COMPLETED",
  "CANCELLED",
  "NO_SHOW",
  "EXPIRED",
  "FAILED",
  "REASSIGNED",
];

const MISSION_LIVE_STATUSES = new Set<DriverMissionStatus>([
  "IN_PROGRESS",
  "ARRIVED",
  "EN_ROUTE",
]);

function normalizeStatus(status: string | null | undefined): DriverMissionStatus | null {
  if (!status || typeof status !== "string") return null;
  const upper = status.trim().toUpperCase() as DriverMissionStatus;
  return upper.length > 0 ? upper : null;
}

function parseScheduledEpoch(mission: DriverMission): number | null {
  const raw = mission.scheduled_time;
  if (raw == null || raw === "") return null;
  const parsed = Date.parse(String(raw));
  return Number.isFinite(parsed) ? parsed : null;
}

/** RDV opérationnel dans les N prochaines minutes (T-30 par défaut). */
export function isOperationalDepartureWithinLeadMinutes(
  mission: DriverMission,
  nowMs: number = Date.now(),
  leadMinutes: number = MISSION_TRACKING_LEAD_MINUTES
): boolean {
  const epoch = parseScheduledEpoch(mission);
  if (epoch == null) return false;
  const leadMs = Math.max(0, leadMinutes) * 60_000;
  const delta = epoch - nowMs;
  return delta <= leadMs;
}

/**
 * Mode tracking pour une mission active (hors fenêtre flotte sans mission).
 * Retourne null si la mission est terminale ou non éligible au tracking mission.
 */
export function resolveMissionTrackingMode(
  mission: DriverMission | null | undefined,
  nowMs: number = Date.now()
): MissionTrackingMode | null {
  if (!mission) return null;
  const status = normalizeStatus(mission.status);
  if (!status) return null;
  if (TRACKING_TERMINAL_STATUSES.includes(status)) return null;

  if (MISSION_LIVE_STATUSES.has(status)) {
    return "mission_live";
  }

  if (status === "ASSIGNED") {
    if (!driverHasScheduledPickupTime(mission)) {
      return "availability_presence";
    }
    if (isOperationalDepartureWithinLeadMinutes(mission, nowMs)) {
      return "mission_live";
    }
    return "availability_presence";
  }

  return null;
}
