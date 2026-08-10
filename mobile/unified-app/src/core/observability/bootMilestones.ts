/**
 * Jalons de démarrage cold start (une émission par processus JS).
 * `DRIVER_MARKERS_MOUNTED` = React a demandé le montage, pas une preuve de paint SDK.
 */

import { emitPerfKpi } from "./perfKpi";

export type BootMilestoneName =
  | "APP_JS_READY"
  | "SESSION_RESTORED"
  | "SESSION_READY"
  | "DASHBOARD_MOUNTED"
  | "DASHBOARD_DATA_READY"
  | "MAP_READY"
  | "SNAPSHOT_DRIVERS_READY"
  | "DRIVER_MARKERS_MOUNTED"
  | "SOCKET_HEALTHY"
  | "DYNAMIC_OVERLAYS_ENABLED";

const coldStartOriginMs = Date.now();
const emitted = new Set<BootMilestoneName>();

export function getBootColdStartOriginMs(): number {
  return coldStartOriginMs;
}

export function resetBootMilestonesForTests(): void {
  emitted.clear();
}

export function markBootMilestone(
  name: BootMilestoneName,
  extra?: Record<string, unknown>
): void {
  if (emitted.has(name)) return;
  emitted.add(name);
  const since_cold_start_ms = Date.now() - coldStartOriginMs;
  emitPerfKpi("perf.boot.milestone", {
    source: "boot.milestones",
    milestone: name,
    since_cold_start_ms,
    ...extra,
  });
}

export function hasBootMilestone(name: BootMilestoneName): boolean {
  return emitted.has(name);
}
