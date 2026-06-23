import type { TrackingSendErrorMeta } from "./driverTrackingSendErrorFormat";

export const CIRCUIT_BREAKER_SUSPEND_MS = 20_000;
export const DEFAULT_RATE_LIMIT_SUSPEND_MS = 60_000;
export const AUTH_SUSPEND_MS = 60_000;
export const FORBIDDEN_SUSPEND_MS = 120_000;

export type QueueSuspendReason =
  | "rate_limit"
  | "auth"
  | "forbidden"
  | "circuit_breaker"
  | "unknown";

export type QueueSuspendState = {
  untilMs: number;
  reason: QueueSuspendReason;
};

export function resolveQueueSuspendMs(
  meta: TrackingSendErrorMeta,
  retryAfterSeconds?: number | null
): { suspendMs: number; reason: QueueSuspendReason } | null {
  if (meta.error_class === "circuit_open") {
    return { suspendMs: CIRCUIT_BREAKER_SUSPEND_MS, reason: "circuit_breaker" };
  }
  if (meta.http_status === 429) {
    const retrySec =
      typeof retryAfterSeconds === "number" && Number.isFinite(retryAfterSeconds)
        ? Math.max(1, Math.floor(retryAfterSeconds))
        : DEFAULT_RATE_LIMIT_SUSPEND_MS / 1000;
    return { suspendMs: retrySec * 1000, reason: "rate_limit" };
  }
  if (meta.http_status === 401) {
    return { suspendMs: AUTH_SUSPEND_MS, reason: "auth" };
  }
  if (meta.http_status === 403) {
    return { suspendMs: FORBIDDEN_SUSPEND_MS, reason: "forbidden" };
  }
  return null;
}
