import type { DriverMission } from "../types";
import { getDriverStatusUx } from "../statusDictionary";
import {
  isOperationalDepartureWithinLeadMinutes,
  resolveMissionTrackingMode,
  TRACKING_TERMINAL_STATUSES,
} from "./resolveMissionTrackingMode";

function normalizeStatus(status: unknown): string {
  return String(status ?? "").trim().toUpperCase();
}

function isTerminalMission(mission: DriverMission): boolean {
  const status = normalizeStatus(mission.status);
  if (TRACKING_TERMINAL_STATUSES.includes(status as (typeof TRACKING_TERMINAL_STATUSES)[number])) {
    return true;
  }
  const ux = getDriverStatusUx(typeof mission.status === "string" ? mission.status : null);
  return ux.terminal;
}

/**
 * Priorité stricte pour le tracking mission :
 * 1 IN_PROGRESS → 2 ARRIVED → 3 EN_ROUTE → 4 ASSIGNED T-30 → 5 ASSIGNED hors fenêtre → exclu si terminal.
 */
export function getMissionTrackingPriority(
  mission: DriverMission,
  nowMs: number = Date.now()
): number {
  const status = normalizeStatus(mission.status);
  if (status === "IN_PROGRESS") return 1;
  if (status === "ARRIVED") return 2;
  if (status === "EN_ROUTE") return 3;
  if (status === "ASSIGNED") {
    if (isOperationalDepartureWithinLeadMinutes(mission, nowMs)) return 4;
    return 5;
  }
  return 99;
}

export function pickTrackingMission(
  missions: DriverMission[] | undefined,
  nowMs: number = Date.now()
): DriverMission | null {
  if (!Array.isArray(missions) || missions.length === 0) return null;

  const candidates = missions.filter((mission) => {
    if (isTerminalMission(mission)) return false;
    return resolveMissionTrackingMode(mission, nowMs) !== null;
  });

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => {
    const priorityDelta = getMissionTrackingPriority(a, nowMs) - getMissionTrackingPriority(b, nowMs);
    if (priorityDelta !== 0) return priorityDelta;
    const aTime = Date.parse(String(a.scheduled_time ?? ""));
    const bTime = Date.parse(String(b.scheduled_time ?? ""));
    const aEpoch = Number.isFinite(aTime) ? aTime : Number.MAX_SAFE_INTEGER;
    const bEpoch = Number.isFinite(bTime) ? bTime : Number.MAX_SAFE_INTEGER;
    return aEpoch - bEpoch;
  });

  return candidates[0] ?? null;
}
