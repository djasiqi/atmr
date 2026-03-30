/**
 * Triggers fermés pour GET /driver/me/bookings/since (header X-LIRIE-Sync-Trigger).
 * Cardinalité fixe — aligné backend (enum + unknown).
 */
export const MISSION_SYNC_TRIGGERS = [
  "socket_connect",
  "foreground",
  "degraded_interval",
  "reconcile_now",
  "reconcile_active",
  "manual_screen",
  "hydrate_empty",
  "socket_booking_event",
] as const;

export type MissionSyncTrigger = (typeof MISSION_SYNC_TRIGGERS)[number];

export const SYNC_TRIGGER_HEADER = "X-LIRIE-Sync-Trigger";

export const SYNC_TRIGGER_UNKNOWN = "unknown";

export function normalizeMissionSyncTrigger(
  raw: string | undefined | null
): MissionSyncTrigger | typeof SYNC_TRIGGER_UNKNOWN {
  if (!raw || typeof raw !== "string") return SYNC_TRIGGER_UNKNOWN;
  const t = raw.trim();
  return (MISSION_SYNC_TRIGGERS as readonly string[]).includes(t)
    ? (t as MissionSyncTrigger)
    : SYNC_TRIGGER_UNKNOWN;
}
