import type { CompanyDriverLiveLocation } from "../../api/contracts";

import { STALE_SECONDS_THRESHOLD } from "../../utils/companyDriverMapStatus";

const STALE_LOCATION_STATUSES = new Set([
  "stale",
  "offline",
  "degraded_constrained",
  "offline_unknown",
]);

const BACKEND_LOCATION_STATUSES = new Set([
  "stale",
  "offline",
  "live",
  "recent",
  "last_known",
  "degraded_constrained",
  "offline_unknown",
]);

/**
 * Signal visuel stale carte — parité web DriverLiveMap (blend gris + opacity).
 * Distinct du statut opérationnel `offline`.
 */
export function isFleetDriverMarkerStale(driver: CompanyDriverLiveLocation): boolean {
  const lastSeenSecondsNumber = Number(driver.last_seen_seconds);
  const locStat = String(
    driver.tracking_display_status || driver.location_status || ""
  ).toLowerCase();
  const hasBackendStatus = BACKEND_LOCATION_STATUSES.has(locStat);
  const staleByAge =
    !driver.location_status &&
    !driver.tracking_display_status &&
    Number.isFinite(lastSeenSecondsNumber) &&
    lastSeenSecondsNumber > STALE_SECONDS_THRESHOLD;
  const staleByStatus = STALE_LOCATION_STATUSES.has(locStat);
  return hasBackendStatus ? staleByStatus : staleByAge;
}
