/**
 * MOB-STARTUP-STORM-FIX-01 — registration FCM idempotente et single-flight.
 * Un rerender React ne doit pas relancer un POST pour le même owner+token.
 */

export type FcmRegistrationAttempt = {
  ownerKey: string;
  token: string;
};

export type FcmRegistrationOutcome = "registered" | "skipped" | "failed";

const FAILURE_BACKOFF_MS = 15_000;

let inFlight: Promise<FcmRegistrationOutcome> | null = null;
let lastSuccessKey: string | null = null;
let lastFailureKey: string | null = null;
let lastFailureAtMs = 0;

export function buildFcmRegistrationKey(ownerKey: string, token: string): string {
  return `${ownerKey}::${token}`;
}

export function getFcmRegistrationInFlightCountForTests(): number {
  return inFlight ? 1 : 0;
}

export function getLastFcmRegistrationSuccessKeyForTests(): string | null {
  return lastSuccessKey;
}

export function resetFcmRegistrationGuardForTests(): void {
  inFlight = null;
  lastSuccessKey = null;
  lastFailureKey = null;
  lastFailureAtMs = 0;
}

/** Invalide le cache si l'owner change (logout / switch chauffeur). */
export function clearFcmRegistrationSuccessIfOwnerChanged(ownerKey: string): void {
  if (!lastSuccessKey) return;
  if (!lastSuccessKey.startsWith(`${ownerKey}::`)) {
    lastSuccessKey = null;
  }
}

/**
 * Au plus 1 registration réseau à la fois.
 * Même owner+token déjà enregistré → skip (0 POST).
 * Échec récent pour la même clé → skip jusqu'au backoff (pas de boucle serrée).
 */
export async function runFcmRegistrationOnce(
  attempt: FcmRegistrationAttempt,
  register: () => Promise<void>
): Promise<FcmRegistrationOutcome> {
  const key = buildFcmRegistrationKey(attempt.ownerKey, attempt.token);
  if (!attempt.ownerKey || !attempt.token) {
    return "skipped";
  }
  if (lastSuccessKey === key) {
    return "skipped";
  }
  if (
    lastFailureKey === key &&
    lastFailureAtMs > 0 &&
    Date.now() - lastFailureAtMs < FAILURE_BACKOFF_MS
  ) {
    return "skipped";
  }
  if (inFlight) {
    return inFlight;
  }
  inFlight = (async (): Promise<FcmRegistrationOutcome> => {
    try {
      await register();
      lastSuccessKey = key;
      lastFailureKey = null;
      lastFailureAtMs = 0;
      return "registered";
    } catch {
      lastFailureKey = key;
      lastFailureAtMs = Date.now();
      return "failed";
    } finally {
      inFlight = null;
    }
  })();
  return inFlight;
}
