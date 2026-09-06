/**
 * OPT-04E — origine des GET /rides (DEV + tests).
 * Pas une campagne de benches : sert à voir un fetch qui ne devrait pas exister.
 */

export type RidesFetchReason =
  | "initial"
  | "pagination"
  | "date_change"
  | "focus"
  | "reconnect"
  | "recovery"
  | "mutation"
  | "manual";

const EMPTY_COUNTS: Record<RidesFetchReason, number> = {
  initial: 0,
  pagination: 0,
  date_change: 0,
  focus: 0,
  reconnect: 0,
  recovery: 0,
  mutation: 0,
  manual: 0,
};

let counts: Record<RidesFetchReason, number> = { ...EMPTY_COUNTS };
let stickyReason: { reason: RidesFetchReason; until: number } | null = null;
let lastAuthoritativeSyncAt = 0;

const STICKY_REASON_TTL_MS = 2_000;
const SKIP_FOCUS_AFTER_SYNC_MS = 8_000;

export function setStickyRidesFetchReason(
  reason: RidesFetchReason,
  ttlMs = STICKY_REASON_TTL_MS
): void {
  stickyReason = { reason, until: Date.now() + ttlMs };
}

export function peekRidesFetchReason(fallback: RidesFetchReason = "initial"): RidesFetchReason {
  if (stickyReason && Date.now() < stickyReason.until) return stickyReason.reason;
  return fallback;
}

export function recordRidesFetch(reason: RidesFetchReason, meta?: Record<string, unknown>): void {
  counts[reason] += 1;
  if (typeof __DEV__ !== "undefined" && __DEV__) {
    console.log("[rides_fetch_reason]", reason, meta ?? {});
  }
}

export function markAuthoritativeDispatchSync(): void {
  lastAuthoritativeSyncAt = Date.now();
}

export function wasRecentlyAuthoritativelySynced(windowMs = SKIP_FOCUS_AFTER_SYNC_MS): boolean {
  return lastAuthoritativeSyncAt > 0 && Date.now() - lastAuthoritativeSyncAt < windowMs;
}

export function getRidesFetchReasonSnapshot(): Record<RidesFetchReason, number> {
  return { ...counts };
}

export function resetRidesFetchReasonForTests(): void {
  counts = { ...EMPTY_COUNTS };
  stickyReason = null;
  lastAuthoritativeSyncAt = 0;
}
