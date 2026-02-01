/**
 * Utilitaires pour normaliser et classifier les statuts de réservation.
 * Garantit que "completed" et "return_completed" sont correctement classés dans "Terminées".
 */

const COMPLETED_STATUSES = [
  'completed',
  'return_completed',
  'return completed', // variante avec espace (legacy)
  'done',
  'finished',
];

const CANCELED_STATUSES = ['canceled', 'cancelled'];

/**
 * Normalise un statut (lowercase, trim).
 * @param {string} [status] - Statut brut
 * @returns {string} Statut normalisé
 */
export function normalizeStatus(status) {
  if (status == null || typeof status !== 'string') return '';
  return String(status).toLowerCase().trim();
}

/**
 * Indique si le statut correspond à une réservation terminée.
 * @param {string} [status] - Statut brut
 * @returns {boolean}
 */
export function isCompletedStatus(status) {
  const norm = normalizeStatus(status);
  return COMPLETED_STATUSES.includes(norm);
}

/**
 * Indique si le statut correspond à une réservation annulée.
 * @param {string} [status] - Statut brut
 * @returns {boolean}
 */
export function isCanceledStatus(status) {
  const norm = normalizeStatus(status);
  return CANCELED_STATUSES.includes(norm);
}

/**
 * Retourne le bucket d'onglet pour un statut (pending, in_progress, completed, canceled).
 * @param {string} [status] - Statut brut
 * @returns {string}
 */
export function getStatusTab(status) {
  const norm = normalizeStatus(status);
  if (COMPLETED_STATUSES.includes(norm)) return 'completed';
  if (CANCELED_STATUSES.includes(norm)) return 'canceled';
  if (['pending'].includes(norm)) return 'pending';
  if (
    ['accepted', 'assigned', 'en_route', 'in_progress'].includes(norm)
  ) {
    return 'in_progress';
  }
  return 'other';
}
