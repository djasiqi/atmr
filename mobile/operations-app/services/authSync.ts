/**
 * Module de synchronisation pour l'authentification
 * 
 * Permet aux intercepteurs de savoir si l'authentification est prête
 * et d'attendre que les tokens soient chargés avant de permettre les requêtes.
 * 
 * Ce module résout la race condition au cold start identifiée dans l'analyse.
 */

import { AuthNotReadyError } from "@/services/authGuards";
import type { AuthLogoutReason } from "@/services/authLogoutReasons";

let authReadyCallback: (() => void) | null = null;
let isAuthReady = false;
export type AuthSessionState =
  | "BOOTSTRAPPING"
  | "READY"
  | "RECOVERING"
  | "DEGRADED"
  | "INVALID";
let authSessionState: AuthSessionState = "BOOTSTRAPPING";
const authStateListeners = new Set<(state: AuthSessionState) => void>();
let waitQueue: Array<{
  resolve: () => void;
  reject: (error: Error) => void;
}> = [];

function emitAuthState(state: AuthSessionState): void {
  authStateListeners.forEach((listener) => {
    try {
      listener(state);
    } catch {
      // no-op
    }
  });
}

export const subscribeAuthSessionState = (
  listener: (state: AuthSessionState) => void
): (() => void) => {
  authStateListeners.add(listener);
  return () => authStateListeners.delete(listener);
};

/**
 * Notifie que l'authentification est prête (appelé depuis useAuth)
 */
export const notifyAuthReady = (): void => {
  isAuthReady = true;
  authSessionState = "READY";
  emitAuthState(authSessionState);
  
  // Résoudre toutes les promesses en attente
  while (waitQueue.length > 0) {
    const { resolve } = waitQueue.shift()!;
    resolve();
  }
  
  // Appeler le callback si défini
  if (authReadyCallback) {
    authReadyCallback();
  }
};

/**
 * Notifie que l'authentification n'est plus prête (logout, reset, etc.)
 */
export const notifyAuthNotReady = (): void => {
  isAuthReady = false;
  authSessionState = "INVALID";
  emitAuthState(authSessionState);
  
  // Rejeter toutes les promesses en attente
  while (waitQueue.length > 0) {
    const { reject } = waitQueue.shift()!;
    reject(new Error("Authentification annulée"));
  }
};

/**
 * Attend que l'authentification soit prête
 *
 * Utilisé par les intercepteurs pour bloquer les requêtes
 * jusqu'à ce que les tokens soient chargés.
 */
export const setAuthStateRecovering = (): void => {
  if (!isAuthReady) {
    authSessionState = "RECOVERING";
    emitAuthState(authSessionState);
  }
};

export const setAuthStateDegraded = (): void => {
  if (!isAuthReady) {
    authSessionState = "DEGRADED";
    emitAuthState(authSessionState);
  }
};

export const setAuthStateBootstrapping = (): void => {
  if (!isAuthReady) {
    authSessionState = "BOOTSTRAPPING";
    emitAuthState(authSessionState);
  }
};

export const getAuthBootstrapState = (): AuthSessionState => authSessionState;

export const assertSessionPurgeAllowed = (
  reason: AuthLogoutReason
): void => {
  const isManualException =
    reason === "manual_logout" || reason === "security_revocation";
  if (authSessionState !== "INVALID" && !isManualException) {
    throw new Error(
      `Session purge forbidden while state=${authSessionState} reason=${reason}`
    );
  }
};

type WaitForAuthReadyOptions = {
  timeoutMs?: number;
  reasonOnTimeout?: "auth_bootstrapping" | "auth_recovering";
};

export const waitForAuthReady = async (
  timeoutOrOptions: number | WaitForAuthReadyOptions = 5000
): Promise<void> => {
  const timeoutMs =
    typeof timeoutOrOptions === "number"
      ? timeoutOrOptions
      : (timeoutOrOptions.timeoutMs ?? 5000);
  const reasonOnTimeout =
    typeof timeoutOrOptions === "number"
      ? "auth_bootstrapping"
      : (timeoutOrOptions.reasonOnTimeout ?? "auth_bootstrapping");
  // Si déjà prêt, résoudre immédiatement
  if (isAuthReady) {
    return Promise.resolve();
  }
  
  // Créer une promesse qui sera résolue quand l'auth sera prête
  return new Promise<void>((resolve, reject) => {
    // Timeout pour éviter d'attendre indéfiniment
    const timeoutId = setTimeout(() => {
      const index = waitQueue.findIndex((item) => item.resolve === resolve);
      if (index !== -1) {
        waitQueue.splice(index, 1);
      }
      authSessionState =
        reasonOnTimeout === "auth_recovering" ? "RECOVERING" : "BOOTSTRAPPING";
      emitAuthState(authSessionState);
      reject(
        new AuthNotReadyError({
          kind: "driver",
          reason: reasonOnTimeout,
        })
      );
    }, timeoutMs);
    
    // Ajouter à la queue
    waitQueue.push({
      resolve: () => {
        clearTimeout(timeoutId);
        resolve();
      },
      reject: (error) => {
        clearTimeout(timeoutId);
        reject(error);
      },
    });
  });
};

/**
 * Vérifie si l'authentification est prête (sans attendre)
 */
export const isAuthReadySync = (): boolean => {
  return isAuthReady;
};

/**
 * Définit un callback qui sera appelé quand l'auth sera prête
 * (utile pour les cas où on veut être notifié une seule fois)
 */
export const setAuthReadyCallback = (callback: (() => void) | null): void => {
  authReadyCallback = callback;
};
