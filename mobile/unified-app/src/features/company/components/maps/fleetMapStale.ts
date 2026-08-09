import type { CompanyDriverLiveLocation } from "../../api/contracts";

import { resolveLocalLocationFreshnessStatus } from "../../utils/localDriverLocationFreshness";

const STALE_LOCATION_STATUSES = new Set([
  "stale",
  "offline",
  "degraded_constrained",
  "offline_unknown",
]);

/**
 * Signal visuel stale carte — âge local depuis recorded_at écrase un `live` serveur figé.
 * Si recorded_at absent/invalide, repli sur le statut serveur explicite.
 */
export function isFleetDriverMarkerStale(driver: CompanyDriverLiveLocation): boolean {
  if (driver.location_status === "last_known") return true;
  const recordedAt = driver.recorded_at ?? driver.timestamp ?? null;
  const localStatus = resolveLocalLocationFreshnessStatus(recordedAt);
  if (recordedAt && localStatus !== "offline_unknown") {
    return localStatus === "stale" || localStatus === "offline";
  }
  const locStat = String(
    driver.tracking_display_status || driver.location_status || ""
  ).toLowerCase();
  return STALE_LOCATION_STATUSES.has(locStat);
}
