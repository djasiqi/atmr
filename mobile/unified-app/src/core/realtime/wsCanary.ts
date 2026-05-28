import { isFeatureEnabled } from "../featureFlags/registry";

const CRITICAL_EVENTS = new Set([
  "booking_updated",
  "booking_cancelled",
  "team_chat_message",
  "dispatch_assignment",
  "dispatch_run_started",
  "dispatch_run_completed",
  "urgent_alert",
]);

/** Headers Traefik pour router vers ws-service (canary). */
export function getWsCanaryExtraHeaders(): Record<string, string> {
  if (!isFeatureEnabled("ws_service_canary")) {
    return {};
  }
  const value = process.env.EXPO_PUBLIC_WS_CANARY_HEADER_VALUE?.trim() || "1";
  return { "X-WS-Canary": value };
}

const ackBuffer: string[] = [];
let ackFlushTimer: ReturnType<typeof setTimeout> | null = null;

function flushAckBatch(socket: { emit: (event: string, data: unknown) => void }) {
  if (ackBuffer.length === 0) return;
  const ids = ackBuffer.splice(0, ackBuffer.length);
  socket.emit("event_ack_batch", { event_ids: ids });
}

/** Ack batch 1–2 s, critical events only (pas GPS). */
export function trackCriticalEventForAck(
  socket: { emit: (event: string, data: unknown) => void },
  eventType: string,
  payload: unknown
): void {
  if (!isFeatureEnabled("ws_service_canary")) return;
  if (!CRITICAL_EVENTS.has(eventType)) return;
  if (!payload || typeof payload !== "object") return;
  const eventId = (payload as { event_id?: string }).event_id;
  if (typeof eventId !== "string" || !eventId) return;
  ackBuffer.push(eventId);
  if (ackFlushTimer) return;
  ackFlushTimer = setTimeout(() => {
    ackFlushTimer = null;
    flushAckBatch(socket);
  }, 1500);
}
