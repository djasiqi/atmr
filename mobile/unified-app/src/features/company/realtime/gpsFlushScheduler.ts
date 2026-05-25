import { MAX_BATCH_AGE_MS, REALTIME_FLUSH_MS } from "./gpsFlushConstants";

export type GpsFlushPriority = "critical" | "visible" | "background";

export const GPS_FLUSH_DELAY_MS: Record<GpsFlushPriority, number> = {
  critical: 0,
  visible: REALTIME_FLUSH_MS,
  background: 800,
};

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
    return priority === "critical" ? 0 : REALTIME_FLUSH_MS;
  }
  return GPS_FLUSH_DELAY_MS[priority];
}

/** maxBatchAge reste global ; les lanes n'augmentent pas la latence max. */
export function getEffectiveMaxBatchAgeMs(): number {
  return MAX_BATCH_AGE_MS;
}
