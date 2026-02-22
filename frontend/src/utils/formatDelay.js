/**
 * Formate un retard en minutes en texte lisible compact.
 * @param {number} minutes - Retard en minutes
 * @returns {string|null} Texte formate ou null si pas de retard
 */
export function formatDelay(minutes) {
  if (!minutes || minutes <= 0) return null;
  if (minutes < 100) return `+${Math.round(minutes)} min`;
  if (minutes < 1440) return `+${Math.round(minutes / 60)} h`;
  return `+${Math.round(minutes / 1440)} j`;
}
