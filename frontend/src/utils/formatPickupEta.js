/**
 * Libellés pour une ETA pickup (ISO 8601 ou Date).
 * @param {string|number|Date|null|undefined} eta
 * @returns {{ text: string; title: string } | null}
 */
export function pickupArrivalHint(eta) {
  if (eta == null || eta === '') return null;
  const t = eta instanceof Date ? eta : new Date(eta);
  if (Number.isNaN(t.getTime())) return null;

  const diffMin = Math.round((t.getTime() - Date.now()) / 60000);
  const clock = t.toLocaleTimeString('fr-CH', { hour: '2-digit', minute: '2-digit' });

  if (diffMin < -5) {
    return {
      text: 'Pickup dépassé',
      title: `ETA ${clock} (${Math.abs(diffMin)} min après l’horaire affichée)`,
    };
  }
  if (diffMin <= 5) {
    return {
      text: 'Arrivée imminente',
      title: `Vers ${clock}`,
    };
  }

  return {
    text: `~${diffMin} min`,
    title: `Arrivée pickup estimée vers ${clock}`,
  };
}
