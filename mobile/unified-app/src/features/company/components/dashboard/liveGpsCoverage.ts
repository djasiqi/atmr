import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { resolveDriverLocationPresence } from "../maps/driverLocationPresence";
import { driverFleetMarkerInitials, resolveDriverDisplayName } from "../../utils/companyDriverMapStatus";

export type GpsCoverageCounts = {
  liveCount: number;
  totalCount: number;
};

export type LiveCoverageRow = {
  driverId: number;
  initials: string;
  name: string;
  isLive: boolean;
  statusLabel: string;
  lastPositionLabel: string | null;
};

/** Compte historique / désactivé : hors dénominateur T. Absence du champ = actif (payload ancien). */
export function isActiveFleetDriver(driver: Pick<CompanyDriverLiveLocation, "is_active">): boolean {
  return driver.is_active !== false;
}

export function toActiveFleetDrivers(
  drivers: CompanyDriverLiveLocation[]
): CompanyDriverLiveLocation[] {
  return drivers.filter(isActiveFleetDriver);
}

function formatLastPositionLabel(ageSeconds: number | null): string | null {
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return null;
  if (ageSeconds < 60) {
    return `Dernière position : il y a ${Math.max(0, Math.floor(ageSeconds))} s`;
  }
  const minutes = Math.max(1, Math.round(ageSeconds / 60));
  if (minutes < 120) return `Dernière position : il y a ${minutes} min`;
  const hours = Math.max(2, Math.round(minutes / 60));
  return `Dernière position : il y a ${hours} h`;
}

/** Compteur N/T — live | recent uniquement, flotte active (pas les comptes historiques). */
export function computeGpsCoverageCounts(
  drivers: CompanyDriverLiveLocation[],
  nowMs: number = Date.now()
): GpsCoverageCounts {
  const fleet = toActiveFleetDrivers(drivers);
  let liveCount = 0;
  for (const driver of fleet) {
    if (resolveDriverLocationPresence(driver, nowMs).countedAsLocated) {
      liveCount += 1;
    }
  }
  return { liveCount, totalCount: fleet.length };
}

export function formatGpsCoverageRatio(liveCount: number, totalCount: number): string {
  return `${liveCount}/${totalCount}`;
}

/** Phrase de synthèse pour la feuille « Suivi en direct ». */
export function formatGpsCoverageSummary(liveCount: number, totalCount: number): string {
  if (totalCount <= 0) return "Aucun chauffeur dans la flotte active";
  if (liveCount <= 0) {
    return `Aucun chauffeur sur ${totalCount} ne transmet actuellement sa position`;
  }
  if (liveCount === 1) {
    return `1 chauffeur sur ${totalCount} transmet actuellement sa position`;
  }
  return `${liveCount} chauffeurs sur ${totalCount} transmettent actuellement leur position`;
}

export function formatGpsCoverageA11y(liveCount: number, totalCount: number): string {
  if (totalCount <= 0) return "Aucun chauffeur dans la flotte active";
  if (liveCount === 1) return `1 chauffeur sur ${totalCount} en direct`;
  return `${liveCount} chauffeurs sur ${totalCount} en direct`;
}

export function buildLiveCoverageRows(
  drivers: CompanyDriverLiveLocation[],
  nowMs: number = Date.now()
): LiveCoverageRow[] {
  const rows = toActiveFleetDrivers(drivers).map((driver) => {
    const presence = resolveDriverLocationPresence(driver, nowMs);
    const isLive = presence.countedAsLocated;
    return {
      driverId: driver.driver_id,
      initials: driverFleetMarkerInitials(driver),
      name: resolveDriverDisplayName(driver),
      isLive,
      statusLabel: isLive ? "En direct" : "Hors ligne",
      lastPositionLabel: isLive ? null : formatLastPositionLabel(presence.ageSeconds),
    };
  });
  return rows.sort((a, b) => {
    if (a.isLive !== b.isLive) return a.isLive ? -1 : 1;
    return a.name.localeCompare(b.name, "fr");
  });
}
