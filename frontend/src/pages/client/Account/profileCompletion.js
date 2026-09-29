/** Score 0–100 selon les champs utiles aux réservations (heuristique côté client). */
export function computeProfileCompletionPercent(profile) {
  if (!profile || typeof profile !== 'object') return 0;
  const t = (v) => String(v ?? '').trim().length > 0;
  let points = 0;
  let max = 0;
  const add = (filled, weight) => {
    max += weight;
    if (filled) points += weight;
  };
  add(t(profile.first_name) && t(profile.last_name), 15);
  add(t(profile.email), 10);
  // Un numéro saisi sans confirmation SMS ne compte pas comme profil terminé.
  add(t(profile.phone) && profile.phone_verified === true, 15);
  add(t(profile.birth_date), 10);
  add(t(profile.gender), 10);
  add(t(profile.address), 25);
  add(t(profile.floor) || t(profile.door_code) || t(profile.access_notes), 15);
  if (max === 0) return 0;
  return Math.min(100, Math.round((points / max) * 100));
}
