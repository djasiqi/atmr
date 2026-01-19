/**
 * Module de synchronisation pour l'authentification
 * 
 * Permet aux intercepteurs de savoir si l'authentification est prête
 * et d'attendre que les tokens soient chargés avant de permettre les requêtes.
 * 
 * Ce module résout la race condition au cold start identifiée dans l'analyse.
 */

let authReadyCallback: (() => void) | null = null;
let isAuthReady = false;
let waitQueue: Array<{
  resolve: () => void;
  reject: (error: Error) => void;
}> = [];

/**
 * Notifie que l'authentification est prête (appelé depuis useAuth)
 */
export const notifyAuthReady = (): void => {
  isAuthReady = true;
  
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
 * 
 * @param timeout Timeout en millisecondes (défaut: 5000ms)
 * @returns Promise qui se résout quand l'auth est prête
 */
export const waitForAuthReady = async (timeout: number = 5000): Promise<void> => {
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
      reject(new Error(`Timeout: authentification non prête après ${timeout}ms`));
    }, timeout);
    
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
