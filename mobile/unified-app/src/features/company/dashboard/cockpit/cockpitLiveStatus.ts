import type { CockpitLiveStatus } from "./cockpitTypes";
import type { CompanyDataFreshness, CompanyTransportStatus } from "../../realtime/companyRealtimeState";

export type CockpitLiveStatusOptions = {
  /** false = pas de WebSocket company (repli HTTP / flag off). */
  realtimeSocketExpected?: boolean;
};

function normalizeTransportStatus(raw: string): CompanyTransportStatus | "unknown" {
  const s = raw.toLowerCase().trim();
  if (s === "idle" || s === "connecting" || s === "healthy" || s === "reconnecting" || s === "failed") {
    return s;
  }
  return "unknown";
}

/**
 * Pill cockpit = transport Socket.IO uniquement (pas la fraîcheur métier).
 * Utiliser `dataFreshness` + `formatCompanyActivityHint` pour l’activité calme / stale.
 */
export function resolveCockpitLiveStatus(
  transportStatus: string,
  options?: CockpitLiveStatusOptions
): CockpitLiveStatus {
  const s = normalizeTransportStatus(transportStatus);
  if (!options?.realtimeSocketExpected && s === "idle") {
    return "http_fallback";
  }
  if (s === "healthy") return "connected";
  if (s === "connecting" || s === "reconnecting") return "reconnecting";
  if (s === "idle") return "initializing";
  return "offline";
}

export function cockpitLiveStatusLabel(
  status: CockpitLiveStatus,
  options?: CockpitLiveStatusOptions
): string {
  if (status === "http_fallback") return "Repli HTTP";
  if (status === "connected") return "LIVE";
  if (status === "reconnecting") return "Reconnexion…";
  if (status === "initializing") return "Connexion…";
  return "Hors ligne";
}

export function cockpitLiveStatusColor(
  status: CockpitLiveStatus,
  options?: CockpitLiveStatusOptions
): string {
  if (status === "http_fallback") return "#64748B";
  if (status === "connected") return "#00796B";
  if (status === "reconnecting") return "#F59E0B";
  if (status === "initializing") return "#94A3B8";
  return "#EF4444";
}

/** Pastille secondaire (orange) quand LIVE + données calmes ou anciennes. */
export function cockpitDataFreshnessAccent(
  liveStatus: CockpitLiveStatus,
  dataFreshness: CompanyDataFreshness
): string | null {
  if (liveStatus !== "connected") return null;
  if (dataFreshness === "stale") return "#F59E0B";
  if (dataFreshness === "idle") return "#94A3B8";
  return null;
}
