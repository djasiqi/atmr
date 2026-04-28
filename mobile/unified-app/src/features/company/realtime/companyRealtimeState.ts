export type CompanyRealtimeStatus =
  | "idle"
  | "connecting"
  | "healthy"
  | "degraded"
  | "reconnecting"
  | "failed";

export type CompanyRealtimeSnapshot = {
  status: CompanyRealtimeStatus;
  connected: boolean;
  contextId: string | null;
  lastEventAt: string | null;
  lastError: string | null;
};

const transitions: Record<CompanyRealtimeStatus, CompanyRealtimeStatus[]> = {
  idle: ["connecting"],
  connecting: ["healthy", "reconnecting", "failed", "degraded"],
  healthy: ["degraded", "reconnecting", "failed"],
  degraded: ["healthy", "reconnecting", "failed"],
  reconnecting: ["healthy", "degraded", "failed"],
  failed: ["connecting", "reconnecting"],
};

export function canTransitionCompanyRealtime(
  from: CompanyRealtimeStatus,
  to: CompanyRealtimeStatus
): boolean {
  return transitions[from]?.includes(to) ?? false;
}

export function reduceCompanyRealtimeStatus(
  current: CompanyRealtimeStatus,
  next: CompanyRealtimeStatus
): CompanyRealtimeStatus {
  if (current === next) return current;
  if (canTransitionCompanyRealtime(current, next)) return next;
  return current;
}

export function getCompanyRealtimePollingIntervalMs(status: CompanyRealtimeStatus): number | null {
  if (status === "healthy") return null;
  if (status === "connecting") return 30_000;
  if (status === "reconnecting") return 15_000;
  if (status === "degraded") return 10_000;
  if (status === "failed") return 10_000;
  return null;
}
