/**
 * Utilitaire de logging pour le mode DEBUG
 * Gère les erreurs réseau silencieusement pour éviter les erreurs dans la console
 */

/**
 * Envoie un log au serveur d'ingestion de manière silencieuse (no-op).
 * @param {Object} _logData - Données du log (ignorées)
 */
export async function sendDebugLog(_logData) {
  // No-op: debug ingest removed
}
