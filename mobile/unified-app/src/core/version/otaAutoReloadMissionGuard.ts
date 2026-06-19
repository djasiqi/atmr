const ACTIVE_MISSION_STATUSES = new Set(["ASSIGNED", "EN_ROUTE", "ARRIVED", "IN_PROGRESS"]);

let missionReloadBlocking = false;

/** Bloque le reload OTA tant qu'une mission chauffeur est en cours. */
export function setOtaAutoReloadMissionBlocking(blocking: boolean): void {
  missionReloadBlocking = blocking;
}

export function isOtaAutoReloadMissionBlocking(): boolean {
  return missionReloadBlocking;
}

export function hasActiveDriverMissionStatus(status: string | null | undefined): boolean {
  if (typeof status !== "string" || status.length === 0) return false;
  return ACTIVE_MISSION_STATUSES.has(status);
}

export function __resetOtaAutoReloadMissionGuardForTests(): void {
  missionReloadBlocking = false;
}
