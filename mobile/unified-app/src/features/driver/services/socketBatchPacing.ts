/**
 * Cadence d'émission socket batch — alignée sur le rate limiter backend
 * (`WS_DRIVER_LOCATION_BATCH_*`). Évite la tempête de re-flush qui provoque
 * des ACK `rate_limited` et une famine du canonical Redis.
 */
const SOCKET_BATCH_MIN_INTERVAL_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_SOCKET_BATCH_MIN_INTERVAL_MS ?? "5000"
);

let lastSocketBatchSentAtMs = 0;
let socketBatchCooldownUntilMs = 0;

export function getSocketBatchMinIntervalMs(): number {
  return SOCKET_BATCH_MIN_INTERVAL_MS;
}

export function getSocketBatchCooldownRemainingMs(nowMs: number = Date.now()): number {
  const pacingUntil = Math.max(
    lastSocketBatchSentAtMs + SOCKET_BATCH_MIN_INTERVAL_MS,
    socketBatchCooldownUntilMs
  );
  return Math.max(0, pacingUntil - nowMs);
}

export function canEmitSocketBatchNow(nowMs: number = Date.now()): boolean {
  return getSocketBatchCooldownRemainingMs(nowMs) <= 0;
}

export function recordSocketBatchSent(nowMs: number = Date.now()): void {
  lastSocketBatchSentAtMs = nowMs;
}

/** Prolonge le cooldown après un rate limit serveur (ACK ou event). */
export function recordSocketBatchRateLimited(
  retryAfterMs: number,
  nowMs: number = Date.now()
): void {
  const bounded = Math.min(15_000, Math.max(SOCKET_BATCH_MIN_INTERVAL_MS, retryAfterMs));
  socketBatchCooldownUntilMs = Math.max(socketBatchCooldownUntilMs, nowMs + bounded);
}

/** Test-only reset */
export function __resetSocketBatchPacingForTests(): void {
  lastSocketBatchSentAtMs = 0;
  socketBatchCooldownUntilMs = 0;
}
