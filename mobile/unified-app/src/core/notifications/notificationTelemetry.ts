import { emitDriverTelemetry } from "../observability/driverTelemetry";

export type NotificationFilterReason =
  | "invalid_payload"
  | "no_active_context"
  | "recipient_role_mismatch"
  | "company_mismatch"
  | "self_actor"
  | "active_screen"
  | "silent"
  | "dedup_hit"
  | string;

export type NotificationPipelineStage =
  | "foreground_handler"
  | "received_listener"
  | "response_listener"
  | "fcm_listener"
  | "initial_response";

function parsePayloadTimestampMs(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const data = payload as Record<string, unknown>;
  const raw =
    data.created_at ??
    data.createdAt ??
    data.sent_at ??
    data.sentAt ??
    data.timestamp;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return raw > 1e12 ? raw : raw * 1000;
  }
  if (typeof raw === "string") {
    const parsed = Date.parse(raw);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function computeNotificationAgeMs(payload: unknown, receivedAtMs = Date.now()): number | null {
  const createdAt = parsePayloadTimestampMs(payload);
  if (createdAt == null) return null;
  return Math.max(0, receivedAtMs - createdAt);
}

export function emitNotificationReceived(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.received", {
    source: "core.notifications.pipeline",
    ...payload,
  });

  void ackPushNotificationReceived(payload).catch(() => {
    // Best-effort : ne pas bloquer le pipeline mobile si l'ack HTTP échoue.
  });
}

async function ackPushNotificationReceived(payload: Record<string, unknown>): Promise<void> {
  const { apiClient } = await import("../api/client");
  await apiClient.post("/driver/me/push-notifications/ack", {
    notification_type: payload.type ?? payload.notification_type,
    booking_id: payload.booking_id ?? payload.mission_id,
    notification_id: payload.event_id ?? payload.trace_id ?? payload.notification_id,
    correlation_id: payload.correlation_id,
    received_at_ms: Date.now(),
  });
}

export function emitNotificationFiltered(
  reason: NotificationFilterReason,
  payload: Record<string, unknown>
): void {
  emitDriverTelemetry("push.notification.filtered", {
    source: "core.notifications.pipeline",
    filter_reason: reason,
    ...payload,
  });
}

export function emitNotificationSuppressed(
  reason: NotificationFilterReason,
  payload: Record<string, unknown>
): void {
  emitDriverTelemetry("push.notification.suppressed", {
    source: "core.notifications.pipeline",
    suppress_reason: reason,
    ...payload,
  });
}

export function emitNotificationDedupHit(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.dedup_hit", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}

export function emitNotificationDuplicateDropped(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.duplicate_dropped", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}

export function emitNotificationProcessingMs(
  durationMs: number,
  payload: Record<string, unknown>
): void {
  emitDriverTelemetry("push.notification.processing_ms", {
    source: "core.notifications.pipeline",
    duration_ms: durationMs,
    ...payload,
  });
}

export function emitNotificationNavigation(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.navigation", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}

export function emitNotificationSoundPlayed(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.sound_played", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}

export function emitNotificationResyncStarted(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.resync_started", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}

export function emitNotificationResyncCompleted(payload: Record<string, unknown>): void {
  emitDriverTelemetry("push.notification.resync_completed", {
    source: "core.notifications.pipeline",
    ...payload,
  });
}
