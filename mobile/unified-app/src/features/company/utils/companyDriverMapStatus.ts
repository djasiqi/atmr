import type { CompanyDriverLiveLocation } from "../api/contracts";

export const STALE_SECONDS_THRESHOLD = 120;

export type CompanyDriverMapCategory = "available" | "en_mission" | "offline";

export function resolveDriverStatus(
  driver: CompanyDriverLiveLocation
): CompanyDriverMapCategory {
  if (isDriverPositionStale(driver)) return "offline";
  if (driver.mission_id != null) return "en_mission";
  return "available";
}

export function isDriverPositionStale(driver: CompanyDriverLiveLocation): boolean {
  const lastSeen = Number(driver.last_seen_seconds);
  const byAge = Number.isFinite(lastSeen) && lastSeen > STALE_SECONDS_THRESHOLD;
  const byStatus = driver.location_status === "stale" || driver.location_status === "offline";
  return byAge || byStatus;
}
