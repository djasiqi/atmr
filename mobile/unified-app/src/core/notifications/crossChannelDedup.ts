import {
  buildNotificationDedupKey,
  markNotificationHandled,
} from "./notificationDedupStore";

/**
 * Retourne true si l'événement a déjà été traité (socket, push ou local).
 */
export function shouldSkipCrossChannelEvent(input: {
  eventId?: string | null;
  notificationId?: string | null;
  missionId?: number | null;
  bookingId?: number | null;
  type?: string | null;
}): boolean {
  const missionId =
    input.missionId ??
    (input.bookingId != null && Number.isFinite(input.bookingId) ? input.bookingId : null);
  const key = buildNotificationDedupKey({
    eventId: input.eventId,
    notificationId: input.notificationId,
    missionId,
    type: input.type,
  });
  return markNotificationHandled(key);
}

export function extractEventIdFromData(data: unknown): string | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  return typeof record.event_id === "string" ? record.event_id : null;
}
