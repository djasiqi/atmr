/**
 * Observabilité tracking — ages GNSS vs runtime + classification déterministe.
 *
 * Chantier P0-C-OBSERVABILITY uniquement : mesure / classification.
 * Aucun effet sur enqueue, flush, ledger ou auth.
 */

export type TrackingObservabilityClass =
  | "HEALTHY"
  | "PIPELINE"
  | "PERSISTENCE"
  | "GNSS"
  | "RUNTIME_ONLY"
  | "NATIVE_RUNTIME"
  | "UNKNOWN";

/** Seuil GNSS stale (s) — aligné historique fix_stale device-health. */
export const LOCATION_FIX_STALE_SECONDS = 300;
/** Seuil task invoke « stale » (s) — distinct du GNSS. */
export const TASK_INVOKE_STALE_SECONDS = 300;
/** Queue considérée bloquée si profondeur > ce seuil. */
export const QUEUE_DEPTH_BLOCKED = 50;
/** Queue bloquée si plus vieux item actif plus âgé que ce seuil (s). */
export const QUEUE_OLDEST_BLOCKED_SECONDS = 120;
/** Persistence stale si lag vs now (ou vs enqueue) dépasse ce seuil (s). */
export const PERSISTENCE_LAG_STALE_SECONDS = 120;

/**
 * Âge GNSS = now - Location.timestamp.
 * Protège timestamps invalides / futurs (null plutôt que faux stale).
 */
export function computeLocationFixAgeSeconds(
  locationTimestampMs: number | null | undefined,
  nowMs: number = Date.now()
): number | null {
  if (typeof locationTimestampMs !== "number" || !Number.isFinite(locationTimestampMs)) {
    return null;
  }
  // Timestamp clairement hors plage (ex. secondes au lieu de ms, NaN) → UNKNOWN
  if (locationTimestampMs < 1_000_000_000_000) {
    // < ~2001 en ms → probablement secondes ; convertir si plausible
    if (locationTimestampMs > 1_000_000_000) {
      locationTimestampMs = locationTimestampMs * 1000;
    } else {
      return null;
    }
  }
  const ageMs = nowMs - locationTimestampMs;
  // Futur > 2 min : invalide (horloge) → ne pas classifier GNSS stale
  if (ageMs < -120_000) {
    return null;
  }
  if (ageMs < 0) {
    return 0;
  }
  return Math.round(ageMs / 1000);
}

/** Âge de la dernière invocation du task natif (≠ GNSS). */
export function computeTaskInvokeAgeSeconds(
  lastTaskInvokedAtMs: number | null | undefined,
  nowMs: number = Date.now()
): number | null {
  if (typeof lastTaskInvokedAtMs !== "number" || !Number.isFinite(lastTaskInvokedAtMs)) {
    return null;
  }
  const ageMs = nowMs - lastTaskInvokedAtMs;
  if (ageMs < -120_000) return null;
  if (ageMs < 0) return 0;
  return Math.round(ageMs / 1000);
}

/** Âge du dernier callback watch JS (≠ Location.timestamp). */
export function computeWatchCallbackAgeSeconds(
  lastWatchAtMs: number | null | undefined,
  nowMs: number = Date.now()
): number | null {
  return computeTaskInvokeAgeSeconds(lastWatchAtMs, nowMs);
}

export type ClassifyTrackingObservabilityInput = {
  locationFixAgeSeconds: number | null;
  taskInvokeAgeSeconds: number | null;
  fgsRunning: boolean;
  fgsExpected: boolean;
  queueDepth: number | null;
  oldestQueueItemAgeSeconds: number | null;
  /** Lag persistence (now - last_persisted_at), null si inconnu. */
  persistenceLagSeconds: number | null;
  /** True si un enqueue récent existe alors que rien n'est persisté. */
  enqueueWithoutPersist?: boolean;
  nowMs?: number;
  fixStaleSeconds?: number;
  taskStaleSeconds?: number;
  queueDepthBlocked?: number;
  queueOldestBlockedSeconds?: number;
  persistenceLagStaleSeconds?: number;
};

function isQueueBlocked(input: ClassifyTrackingObservabilityInput): boolean {
  const depthBlocked = input.queueDepthBlocked ?? QUEUE_DEPTH_BLOCKED;
  const oldestBlocked = input.queueOldestBlockedSeconds ?? QUEUE_OLDEST_BLOCKED_SECONDS;
  if ((input.queueDepth ?? 0) > depthBlocked) return true;
  if (
    input.oldestQueueItemAgeSeconds != null
    && input.oldestQueueItemAgeSeconds > oldestBlocked
  ) {
    return true;
  }
  return false;
}

function isPersistenceStale(input: ClassifyTrackingObservabilityInput): boolean {
  const lagStale = input.persistenceLagStaleSeconds ?? PERSISTENCE_LAG_STALE_SECONDS;
  if (input.persistenceLagSeconds != null && input.persistenceLagSeconds > lagStale) {
    return true;
  }
  if (input.enqueueWithoutPersist === true) {
    return true;
  }
  return false;
}

/**
 * Classification déterministe (priorité) :
 * UNKNOWN → GNSS → PIPELINE → PERSISTENCE → RUNTIME_ONLY → NATIVE_RUNTIME → HEALTHY
 *
 * Invariant : task stale + fix fresh ≠ GPS stale (RUNTIME_ONLY).
 */
export function classifyTrackingObservability(
  input: ClassifyTrackingObservabilityInput
): TrackingObservabilityClass {
  const fixStale = input.fixStaleSeconds ?? LOCATION_FIX_STALE_SECONDS;
  const taskStale = input.taskStaleSeconds ?? TASK_INVOKE_STALE_SECONDS;

  if (input.locationFixAgeSeconds == null) {
    return "UNKNOWN";
  }

  if (input.locationFixAgeSeconds > fixStale) {
    return "GNSS";
  }

  // Fix frais (ou ≤ seuil)
  if (isQueueBlocked(input)) {
    return "PIPELINE";
  }

  if (isPersistenceStale(input)) {
    return "PERSISTENCE";
  }

  if (
    input.taskInvokeAgeSeconds != null
    && input.taskInvokeAgeSeconds > taskStale
  ) {
    return "RUNTIME_ONLY";
  }

  if (input.fgsExpected && !input.fgsRunning) {
    return "NATIVE_RUNTIME";
  }

  return "HEALTHY";
}

/** True seulement si la classe est GNSS (base d'alerte GPS stale). */
export function isGpsStaleAlertClass(
  observabilityClass: TrackingObservabilityClass
): boolean {
  return observabilityClass === "GNSS";
}
