/**
 * Utilitaire de logging pour le mode DEBUG
 * Gère les erreurs réseau silencieusement pour éviter les erreurs dans la console
 */

// Désactiver les logs si la variable d'environnement est définie
const DEBUG_LOGGING_ENABLED = process.env.REACT_APP_DEBUG_LOGGING !== 'false';

let serverAccessible = null; // null = pas encore vérifié, true = accessible, false = inaccessible
let checkInProgress = false;
let lastCheckTime = 0;
const CHECK_INTERVAL = 30000; // Check every 30 seconds
let failedAttempts = 0;
const MAX_FAILED_ATTEMPTS = 3; // Désactiver après 3 tentatives échouées

// Vérification initiale au chargement du module
if (typeof window !== 'undefined' && DEBUG_LOGGING_ENABLED) {
  // Faire une vérification initiale après un court délai pour éviter de bloquer le chargement
  setTimeout(() => {
    checkServerAccessibility().catch(() => {
      // Ignorer les erreurs de la vérification initiale
    });
  }, 1000);
}

/**
 * Vérifie si le serveur d'ingestion est accessible
 * @returns {Promise<boolean>}
 */
async function checkServerAccessibility() {
  if (checkInProgress) return serverAccessible;
  
  const now = Date.now();
  if (now - lastCheckTime < CHECK_INTERVAL && lastCheckTime > 0) {
    return serverAccessible;
  }
  
  checkInProgress = true;
  lastCheckTime = now;
  
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1000);
    
    const response = await fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test: 'accessibility_check' }),
      signal: controller.signal,
      credentials: 'omit', // Ne pas envoyer de credentials pour éviter les problèmes CORS
    });
    
    clearTimeout(timeoutId);
    serverAccessible = response.ok || response.status < 500;
  } catch (error) {
    // Silently mark server as inaccessible
    serverAccessible = false;
  } finally {
    checkInProgress = false;
  }
  
  return serverAccessible;
}

/**
 * Envoie un log au serveur d'ingestion de manière silencieuse
 * @param {Object} logData - Données du log
 */
export async function sendDebugLog(logData) {
  // Si le logging est désactivé, ne rien faire
  if (!DEBUG_LOGGING_ENABLED) {
    return;
  }
  
  // Si le serveur a échoué trop de fois, ne plus essayer
  if (failedAttempts >= MAX_FAILED_ATTEMPTS) {
    return;
  }
  
  // Si le serveur est connu comme inaccessible, ne pas essayer
  if (serverAccessible === false && Date.now() - lastCheckTime < CHECK_INTERVAL) {
    return;
  }
  
  // Si on n'a pas encore vérifié ou si on doit re-vérifier, faire une vérification
  if (serverAccessible === null || (serverAccessible === false && Date.now() - lastCheckTime >= CHECK_INTERVAL)) {
    const isAccessible = await checkServerAccessibility();
    if (!isAccessible) {
      failedAttempts++;
      return;
    }
  }
  
  // Si on sait que le serveur n'est pas accessible, ne pas essayer
  if (serverAccessible === false) {
    return;
  }
  
  // Réinitialiser le compteur d'échecs si le serveur est accessible
  if (serverAccessible === true && failedAttempts > 0) {
    failedAttempts = 0;
  }
  
  // Envoyer le log avec gestion d'erreur silencieuse
  // Utiliser fetch avec credentials: 'omit' pour éviter les problèmes CORS
  // (sendBeacon envoie toujours les credentials et ne peut pas être désactivé)
  try {
    const logDataStr = JSON.stringify(logData);
    
    // Utiliser fetch avec AbortController pour éviter les timeouts longs
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    
    await fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: logDataStr,
      signal: controller.signal,
      credentials: 'omit', // Ne pas envoyer de credentials pour éviter les problèmes CORS
    });
    
    clearTimeout(timeoutId);
    failedAttempts = 0; // Réinitialiser le compteur en cas de succès
  } catch (error) {
    // Erreur silencieuse - ne pas logger dans la console
    // Marquer le serveur comme inaccessible pour éviter d'autres tentatives
    if (error.name !== 'AbortError') {
      failedAttempts++;
      serverAccessible = false;
    }
  }
}

