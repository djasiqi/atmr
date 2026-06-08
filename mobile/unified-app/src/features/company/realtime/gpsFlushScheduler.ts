import { resolveMaxBatchAgeMs, resolveRealtimeFlushMs } from "./companyMapRuntimeConfig";

export type GpsFlushPriority = "critical" | "visible" | "background";

export function gpsFlushDelayMsForPriority(priority: GpsFlushPriority): number {
  if (priority === "critical") return 0;
  if (priority === "background") return 800;
  return resolveRealtimeFlushMs();
}

/** Priorité d'un événement temps réel carte (lanes optionnelles Phase 3). */
export function resolveGpsEventFlushPriority(
  eventType: string,
  options?: { immediate?: boolean; observabilityOnly?: boolean }
): GpsFlushPriority {
  if (options?.immediate) return "critical";
  if (eventType === "company_socket_reconnected") return "critical";
  if (eventType === "driver_live_state_update") return "critical";
  if (options?.observabilityOnly) return "background";
  return "visible";
}

export function resolveFlushDelayMs(
  priority: GpsFlushPriority,
  lanesEnabled: boolean
): number {
  if (!lanesEnabled) {
    return priority === "critical" ? 0 : resolveRealtimeFlushMs();
  }
  return gpsFlushDelayMsForPriority(priority);
}

/** maxBatchAge reste global ; les lanes n'augmentent pas la latence max. */
export function getEffectiveMaxBatchAgeMs(): number {
  return resolveMaxBatchAgeMs();
}
