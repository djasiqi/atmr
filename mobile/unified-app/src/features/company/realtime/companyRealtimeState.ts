/** Santé de la connexion Socket.IO (transport uniquement). */
export type CompanyTransportStatus =
  | "idle"
  | "connecting"
  | "healthy"
  | "reconnecting"
  | "failed";

/** Alias historique — même sémantique que `CompanyTransportStatus`. */
export type CompanyRealtimeStatus = CompanyTransportStatus;

/** Fraîcheur des événements métier (indépendante du transport). */
export type CompanyDataFreshness = "fresh" | "idle" | "stale";

/** Activité récente (< 60 s depuis le dernier événement métier). */
export const COMPANY_DATA_FRESH_MS = 60_000;
/** Calme : pas d’événement depuis 60 s à 5 min. */
export const COMPANY_DATA_STALE_MS = 5 * 60_000;

export type CompanyRealtimeSnapshot = {
  /** Transport — alias de `transportStatus` (rétrocompat). */
  status: CompanyTransportStatus;
  transportStatus: CompanyTransportStatus;
  dataFreshness: CompanyDataFreshness;
  connected: boolean;
  contextId: string | null;
  lastEventAt: string | null;
  lastConnectedAt: string | null;
  lastError: string | null;
};

const transportTransitions: Record<CompanyTransportStatus, CompanyTransportStatus[]> = {
  idle: ["connecting", "reconnecting", "failed"],
  connecting: ["healthy", "reconnecting", "failed"],
  healthy: ["reconnecting", "failed"],
  reconnecting: ["healthy", "failed"],
  failed: ["connecting", "reconnecting"],
};

export function canTransitionCompanyTransport(
  from: CompanyTransportStatus,
  to: CompanyTransportStatus
): boolean {
  return transportTransitions[from]?.includes(to) ?? false;
}

/** @deprecated utiliser `canTransitionCompanyTransport` */
export const canTransitionCompanyRealtime = canTransitionCompanyTransport;

export function reduceCompanyTransportStatus(
  current: CompanyTransportStatus,
  next: CompanyTransportStatus
): CompanyTransportStatus {
  if (current === next) return current;
  if (canTransitionCompanyTransport(current, next)) return next;
  return current;
}

/** @deprecated utiliser `reduceCompanyTransportStatus` */
export const reduceCompanyRealtimeStatus = reduceCompanyTransportStatus;

export function computeCompanyDataFreshness(
  lastEventAt: string | null,
  nowMs: number = Date.now()
): CompanyDataFreshness {
  if (!lastEventAt) return "idle";
  const age = nowMs - Date.parse(lastEventAt);
  if (!Number.isFinite(age) || age < 0) return "idle";
  if (age < COMPANY_DATA_FRESH_MS) return "fresh";
  if (age < COMPANY_DATA_STALE_MS) return "idle";
  return "stale";
}

export function normalizeCompanyRealtimeSnapshot(
  partial: Partial<CompanyRealtimeSnapshot> & {
    status?: CompanyTransportStatus;
    transportStatus?: CompanyTransportStatus;
  }
): CompanyRealtimeSnapshot {
  const transport = partial.transportStatus ?? partial.status ?? "idle";
  const lastEventAt = partial.lastEventAt ?? null;
  return {
    status: transport,
    transportStatus: transport,
    dataFreshness:
      partial.dataFreshness ?? computeCompanyDataFreshness(lastEventAt),
    connected: partial.connected ?? false,
    contextId: partial.contextId ?? null,
    lastEventAt,
    lastConnectedAt: partial.lastConnectedAt ?? null,
    lastError: partial.lastError ?? null,
  };
}

export function getCompanyRealtimePollingIntervalMs(
  transportStatus: CompanyTransportStatus,
  dataFreshness: CompanyDataFreshness = "idle"
): number | null {
  if (transportStatus === "healthy") {
    return dataFreshness === "stale" ? 10_000 : null;
  }
  if (transportStatus === "connecting") return 30_000;
  if (transportStatus === "reconnecting") return 15_000;
  if (transportStatus === "failed") return 10_000;
  return null;
}
