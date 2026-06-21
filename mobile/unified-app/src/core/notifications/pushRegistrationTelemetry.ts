import { Platform } from "react-native";

import { emitDriverTelemetry } from "../observability/driverTelemetry";

export type PushRegistrationTelemetryEvent =
  | "driver_push.bridge_mounted"
  | "driver_push.disclosure_blocked"
  | "driver_push.permission_blocked"
  | "driver_push.get_token_failed"
  | "driver_push.token_acquired"
  | "driver_push.register_success";

export type PushRegistrationTelemetryPayload = {
  source: string;
  stage?: string | null;
  reason?: string | null;
  provider?: "expo" | "fcm" | null;
  token_length?: number | null;
  enabled?: boolean;
  fcm_enabled?: boolean;
  driver_id?: number | null;
  context_type?: string | null;
  error_code?: string | null;
  permission_status?: string | null;
};

function getApiClient(): { post?: (url: string, body: unknown) => Promise<unknown> } | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("../api/client") as {
      apiClient?: { post?: (url: string, body: unknown) => Promise<unknown> };
    };
    return mod.apiClient ?? null;
  } catch {
    return null;
  }
}

/**
 * Télémétrie push observable en prod : POST /driver/me/telemetry/push (logs backend).
 * Complète emitDriverTelemetry (ingest local __DEV__ uniquement).
 */
export function reportPushRegistrationTelemetry(
  event: PushRegistrationTelemetryEvent,
  payload: PushRegistrationTelemetryPayload
): void {
  emitDriverTelemetry(event, payload);

  const client = getApiClient();
  if (!client || typeof client.post !== "function") return;

  void client
    .post("/driver/me/telemetry/push", {
      event,
      platform: Platform.OS,
      ...payload,
    })
    .catch(() => undefined);
}
