import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";

export type SilentPushPayload = {
  type?: string;
  mission_id?: number | string;
  missionId?: number | string;
  booking_id?: number | string;
  silent?: boolean | number | string;
  silent_push?: boolean | number | string;
  background?: boolean | number | string;
  content_available?: boolean | number | string;
};

type VisualPushPayload = {
  type?: string;
  title?: unknown;
  body?: unknown;
  notification?: {
    title?: unknown;
    body?: unknown;
  };
};

function parseMissionId(input: SilentPushPayload): number | null {
  const raw = input.mission_id ?? input.missionId ?? input.booking_id;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function isSilentPayload(input: unknown): input is SilentPushPayload {
  if (!input || typeof input !== "object") return false;
  const value = input as SilentPushPayload;
  const rawType = typeof value.type === "string" ? value.type.toLowerCase() : "";
  if (rawType === "mission_refresh" || rawType === "silent_update") return true;
  const silent = value.silent ?? value.silent_push ?? value.background ?? value.content_available;
  if (typeof silent === "boolean") return silent;
  if (typeof silent === "number") return silent === 1;
  if (typeof silent === "string") {
    const normalized = silent.toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "silent";
  }
  return false;
}

function hasNonEmptyText(value: unknown): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

export function shouldSuppressVisualPush(input: unknown): boolean {
  if (!input || typeof input !== "object") return false;
  const payload = input as VisualPushPayload;
  const rawType = typeof payload.type === "string" ? payload.type.toLowerCase() : "";
  if (rawType !== "mission_updated" && rawType !== "booking_updated") {
    return false;
  }

  const hasTitle = hasNonEmptyText(payload.title) || hasNonEmptyText(payload.notification?.title);
  const hasBody = hasNonEmptyText(payload.body) || hasNonEmptyText(payload.notification?.body);
  // Data-only mission_updated payloads are usually technical sync events.
  // Avoid noisy fallback notifications ("Mise à jour mission").
  return !hasTitle && !hasBody;
}

export async function handleSilentPushPayload(
  payload: unknown,
  onResync: (missionId: number | null) => Promise<void>
): Promise<void> {
  if (!isSilentPayload(payload)) return;
  const missionId = parseMissionId(payload as SilentPushPayload);
  emitDriverTelemetry("push.notification.silent_sync", {
    source: "driver.silent_notifications",
    mission_id: missionId,
  });
  await onResync(missionId);
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const bgTask = require("./services/backgroundLocationTask") as typeof import("./services/backgroundLocationTask");
    if (typeof bgTask.restartNativeTrackingFromWake === "function") {
      await bgTask.restartNativeTrackingFromWake("silent_push_payload");
    }
    const client = require("../../core/api/client") as { apiClient?: { post?: Function } };
    await client.apiClient?.post?.("/driver/me/push-notifications/silent-ack", {
      sync_type: (payload as SilentPushPayload)?.type || "silent_update",
      result: "acked",
    });
  } catch {
    /* noop */
  }
}

/** Branche le handler silent push par défaut (wake + ack) avant montage NotificationsProvider. */
export function registerDefaultSilentPushBackgroundHandler(): void {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const fcm = require("./firebaseMessaging") as typeof import("./firebaseMessaging");
  fcm.setDriverFcmBackgroundCallback(async (payload) => {
    await handleSilentPushPayload(payload, async () => {
      /* resync complet branché par NotificationsProvider au foreground */
    });
  });
}
