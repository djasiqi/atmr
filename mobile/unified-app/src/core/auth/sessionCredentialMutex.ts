/**
 * Verrous credentials session (PR2).
 * - withCredentialStoreLock : données historiques intersessions (PendingRevocation, …)
 * - withSessionCredentialMutation : données de la session courante (+ expectedGeneration)
 * - claimNextSessionGenerationIfCurrent : compare-and-bump atomique
 */
import {
  bumpSessionGeneration,
  isCurrentSessionGeneration,
  type SessionGenerationId,
} from "./authCredentialStore";

let storeLockTail: Promise<void> = Promise.resolve();

/**
 * Sérialise les mutations SecureStore / pending historiques.
 * Aucun réseau ni GPS sous ce verrou.
 */
export async function withCredentialStoreLock<T>(
  mutation: () => Promise<T> | T
): Promise<T> {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const previous = storeLockTail;
  storeLockTail = previous.then(() => gate).catch(() => gate);
  await previous.catch(() => undefined);
  try {
    return await mutation();
  } finally {
    release();
  }
}

export type SessionCredentialMutationResult<T> =
  | { status: "applied"; value: T }
  | { status: "stale" };

/**
 * Mutex session : store lock + vérification de génération fournie par l'appelant.
 * Ne fournit jamais la génération courante — l'appelant doit la capturer/claimer.
 */
export async function withSessionCredentialMutation<T>(
  expectedGeneration: SessionGenerationId,
  mutation: () => Promise<T> | T
): Promise<SessionCredentialMutationResult<T>> {
  return withCredentialStoreLock(async () => {
    if (!isCurrentSessionGeneration(expectedGeneration)) {
      return { status: "stale" };
    }
    const value = await mutation();
    if (!isCurrentSessionGeneration(expectedGeneration)) {
      return { status: "stale" };
    }
    return { status: "applied", value };
  });
}

export type ClaimNextGenerationResult =
  | { status: "claimed"; generation: SessionGenerationId }
  | { status: "stale" };

/**
 * Compare-and-bump atomique sous le même verrou :
 * vérifie expectedGeneration puis incrémente et retourne la nouvelle génération.
 */
export async function claimNextSessionGenerationIfCurrent(
  expectedGeneration: SessionGenerationId
): Promise<ClaimNextGenerationResult> {
  return withCredentialStoreLock(() => {
    if (!isCurrentSessionGeneration(expectedGeneration)) {
      return { status: "stale" };
    }
    const generation = bumpSessionGeneration();
    return { status: "claimed", generation };
  });
}

/** Réservé aux tests — vide la file de verrou (process isolé Jest). */
export function __resetCredentialStoreLockForTests(): void {
  storeLockTail = Promise.resolve();
}
