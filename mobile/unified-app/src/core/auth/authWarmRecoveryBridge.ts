/**
 * Pont P0-5 : une seule entrée warm recovery (API 401, socket, foreground).
 * Si SessionProvider a enregistré son handler, on l'utilise (statut + purge génération).
 * Sinon fallback sur attemptRestRecovery (single-flight nu).
 */
import type { RecoveryOutcome } from "./authRecoveryCoordinator";

type WarmHandler = (reason: string) => Promise<RecoveryOutcome>;

let warmHandler: WarmHandler | null = null;

export function setWarmAuthRecoveryHandler(handler: WarmHandler | null): () => void {
  warmHandler = handler;
  return () => {
    if (warmHandler === handler) {
      warmHandler = null;
    }
  };
}

export async function requestWarmAuthRecovery(reason: string): Promise<RecoveryOutcome> {
  if (warmHandler) {
    return warmHandler(reason);
  }
  const { attemptRestRecovery } = require("./authRecoveryCoordinator") as {
    attemptRestRecovery: (r: string) => Promise<RecoveryOutcome>;
  };
  return attemptRestRecovery(reason);
}
