import type { CompanyDriverLiveLocation } from "../api/contracts";

export const STALE_SECONDS_THRESHOLD = 120;

/** Diamètre pastille chauffeur carte flotte (identique natif et variante navigateur). */
export const FLEET_DRIVER_MARKER_DISC_DP = 28;

export type CompanyDriverMapCategory = "available" | "en_mission" | "last_known" | "offline";

import { FLEET_WEB_STATUS_COLORS } from "../components/maps/fleetMapStatusContract";

/** Pastilles carte flotte — alignées sur la charte Lirie web. */
export const DRIVER_FLEET_MARKER_PALETTE: Record<
  CompanyDriverMapCategory,
  { fill: string; pinScale: number; label: string }
> = {
  available: { fill: FLEET_WEB_STATUS_COLORS.available, pinScale: 1.05, label: "Disponible" },
  en_mission: { fill: FLEET_WEB_STATUS_COLORS.busy, pinScale: 1, label: "En mission" },
  last_known: { fill: FLEET_WEB_STATUS_COLORS.offline, pinScale: 0.95, label: "Dernière position connue" },
  offline: { fill: FLEET_WEB_STATUS_COLORS.offline, pinScale: 0.9, label: "Position périmée ou hors ligne" },
};

export function resolveDriverStatus(
  driver: CompanyDriverLiveLocation,
  options?: { hasActiveMission?: boolean }
): CompanyDriverMapCategory {
  if (driver.location_status === "last_known") return "last_known";
  if (isDriverPositionStale(driver)) return "offline";
  if (options?.hasActiveMission === false) return "available";
  if (driver.mission_id != null) return "en_mission";
  return "available";
}

function cleanName(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const t = value.trim();
  return t.length > 0 ? t : null;
}

export type DriverDisplayNameOptions = {
  /** Nom chauffeur issu de la mission liée (API dispatch / transfert). */
  missionDriverName?: string | null;
  /** Nom de l'entreprise active (ex. « Emmenez-moi ») si le chauffeur n'a pas d'identité affichable. */
  organizationName?: string | null;
};

export function resolveDriverDisplayName(
  driver: CompanyDriverLiveLocation,
  options?: DriverDisplayNameOptions
): string {
  const full = cleanName(driver.full_name);
  if (full) return full;
  const first = cleanName(driver.first_name);
  const last = cleanName(driver.last_name);
  const merged = [first, last].filter(Boolean).join(" ").trim();
  if (merged) return merged;
  const driverName = cleanName(driver.driver_name);
  if (driverName) return driverName;
  const missionName = cleanName(options?.missionDriverName);
  if (missionName) return missionName;
  const org = cleanName(options?.organizationName);
  if (org) return org;
  return `Chauffeur #${driver.driver_id}`;
}

/** Première lettre alphabétique d’un mot (accents conservés). */
export function pickFleetMarkerWordInitial(word: string): string {
  const letters = word.replace(/[^A-Za-zÀ-ÖØ-öø-ÿ]/g, "");
  const initial = letters[0] ?? word[0] ?? "";
  return initial.toUpperCase();
}

/**
 * Initiales marqueur — parité web `getDriverMarkerLabel` (DriverLiveMap.jsx).
 */
export function resolveFleetMarkerInitialsFromDisplayName(fullName: string): string {
  const words = fullName.trim().split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    return `${pickFleetMarkerWordInitial(words[0])}${pickFleetMarkerWordInitial(words[1])}`;
  }
  return fullName.trim().slice(0, 2).toUpperCase();
}

export function driverFleetMarkerInitials(driver: CompanyDriverLiveLocation): string {
  const first = cleanName(driver.first_name);
  const last = cleanName(driver.last_name);
  if (first && last) {
    return `${pickFleetMarkerWordInitial(first)}${pickFleetMarkerWordInitial(last)}`;
  }
  const full = cleanName(driver.full_name);
  if (full) {
    const fromFull = resolveFleetMarkerInitialsFromDisplayName(full);
    const wordCount = full.trim().split(/\s+/).filter(Boolean).length;
    if (wordCount >= 2) return fromFull;
  }
  return resolveFleetMarkerInitialsFromDisplayName(resolveDriverDisplayName(driver));
}

export function driverFleetMarkerTitle(driver: CompanyDriverLiveLocation): string {
  return resolveDriverDisplayName(driver);
}

export function driverFleetMarkerDescription(driver: CompanyDriverLiveLocation): string {
  const cat = resolveDriverStatus(driver);
  const parts = [DRIVER_FLEET_MARKER_PALETTE[cat].label];
  if (driver.mission_id != null) {
    parts.push(`Mission #${driver.mission_id}`);
  }
  parts.push(`ID #${driver.driver_id}`);
  return parts.join(" · ");
}

export function isDriverPositionStale(driver: CompanyDriverLiveLocation): boolean {
  if (driver.location_status === "last_known") return false;
  const lastSeen = Number(driver.last_seen_seconds);
  const byAge = Number.isFinite(lastSeen) && lastSeen > STALE_SECONDS_THRESHOLD;
  const byStatus = driver.location_status === "stale" || driver.location_status === "offline";
  return byAge || byStatus;
}
