import type { DriverTelemetryEventName, DriverTelemetryPayload } from "./driverTelemetry";
import { emitDriverTelemetry } from "./driverTelemetry";

export type PlatformTelemetryContext = "driver" | "client" | "company" | "institution";

export type PlatformTelemetryPayload = {
  source: string;
  context_id?: string | null;
  user_id?: string | null;
  mission_id?: number | null;
  app_state?: string | null;
  network_state?: string | null;
  [key: string]: unknown;
};

const driverEventNames = new Set<DriverTelemetryEventName>([
  "auth.refresh.failure",
  "auth.bootstrap.failure",
  "realtime.socket.disconnect",
  "realtime.socket.reconnect",
  "tracking.permission.denied",
  "tracking.send.backoff",
  "tracking.send.failure",
  "tracking.send.recovered",
  "transition.queue.retry",
  "transition.queue.flush",
  "transition.queue.failure",
  "push.token.registered",
  "push.notification.received",
  "push.notification.opened",
  "driver.runtime.heartbeat",
]);

function isDriverTelemetryEvent(event: string): event is DriverTelemetryEventName {
  return driverEventNames.has(event as DriverTelemetryEventName);
}

export function emitPlatformTelemetry(
  context: PlatformTelemetryContext,
  event: string,
  payload: PlatformTelemetryPayload
) {
  if (context === "driver" && isDriverTelemetryEvent(event)) {
    emitDriverTelemetry(event, payload as DriverTelemetryPayload);
    return;
  }

  console.info(`[platform-telemetry] ${context}.${event}`, payload);
}
