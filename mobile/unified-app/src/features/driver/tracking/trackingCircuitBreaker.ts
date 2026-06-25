export type CircuitState = "closed" | "open" | "half_open";

const OPEN_THRESHOLD = 3;
const OPEN_COOLDOWN_MS = 60_000;

export type TrackingCircuitBreakerSnapshot = {
  state: CircuitState;
  consecutiveFailures: number;
  openedAtMs: number | null;
};

let consecutiveFailures = 0;
let state: CircuitState = "closed";
let openedAtMs: number | null = null;

export function getTrackingCircuitBreakerSnapshot(): TrackingCircuitBreakerSnapshot {
  return { state, consecutiveFailures, openedAtMs };
}

export function resetTrackingCircuitBreaker(): void {
  consecutiveFailures = 0;
  state = "closed";
  openedAtMs = null;
}

/** Enregistre un échec (ex. timeout GPS). Retourne true si le circuit vient de s'ouvrir. */
export function recordTrackingCircuitFailure(nowMs: number = Date.now()): boolean {
  consecutiveFailures += 1;
  if (consecutiveFailures >= OPEN_THRESHOLD && state === "closed") {
    state = "open";
    openedAtMs = nowMs;
    return true;
  }
  return false;
}

/** Succès — repasse en closed ou half_open → closed. */
export function recordTrackingCircuitSuccess(): void {
  consecutiveFailures = 0;
  state = "closed";
  openedAtMs = null;
}

/** Peut tenter une opération (capture/envoi) ? Circuit OPEN bloque sauf recovery. */
export function canAttemptTrackingOperation(
  nowMs: number = Date.now(),
  allowRecovery = false
): boolean {
  if (state === "closed" || state === "half_open") {
    return true;
  }
  if (state === "open" && openedAtMs != null && nowMs - openedAtMs >= OPEN_COOLDOWN_MS) {
    state = "half_open";
    return allowRecovery;
  }
  return allowRecovery;
}

export function isTrackingCircuitOpen(): boolean {
  return state === "open";
}
