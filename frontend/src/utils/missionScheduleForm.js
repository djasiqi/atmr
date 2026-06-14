/**
 * Helpers formulaire institution — date mission + heures par étape.
 * Règle : heure saisie ⇒ confirmée ; heure vide ⇒ non confirmée.
 */

export const normalizeMissionDate = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const match = raw.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (match) {
    const [, day, month, year] = match;
    return `${year}-${month}-${day}`;
  }
  return raw;
};

export const combineMissionDateTime = (missionDate, timeHHMM) => {
  const normalizedDate = normalizeMissionDate(missionDate);
  if (!normalizedDate || !timeHHMM?.trim()) return null;
  const d = new Date(`${normalizedDate}T${timeHHMM.trim()}:00`);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
};

/** Délai minimal (minutes) entre maintenant et un rendez-vous / arrivée. */
export const MIN_ARRIVAL_LEAD_MINUTES = 60;

/** Tolérance (minutes) pour le départ afin d'accepter « maintenant » (raccourci Urgent). */
const PICKUP_PAST_GRACE_MINUTES = 2;

/** True si l'instant ISO est dans le passé (au-delà de la tolérance). */
export const isInstantInPast = (iso, graceMinutes = PICKUP_PAST_GRACE_MINUTES) => {
  if (!iso) return false;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return false;
  return t < Date.now() - graceMinutes * 60000;
};

/** True si l'instant ISO est avant maintenant + leadMinutes (rendez-vous trop proche/passé). */
export const isInstantBeforeLead = (iso, leadMinutes = MIN_ARRIVAL_LEAD_MINUTES) => {
  if (!iso) return false;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return false;
  return t < Date.now() + leadMinutes * 60000;
};

/** Normalise une valeur d'heure (ISO, "YYYY-MM-DDTHH:MM" ou "HH:MM") en "HH:MM". */
export const extractHHMM = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  if (/^\d{2}:\d{2}$/.test(raw)) return raw;
  const d = new Date(raw);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  }
  const part = raw.split('T')[1];
  return part ? part.slice(0, 5) : '';
};

/** Dérive pickup_time_confirmed depuis la présence d'une heure de départ. */
export const derivePickupTimeConfirmed = (pickupTime) => Boolean(pickupTime?.trim());

/**
 * Applique le départ au payload : pickup confirmé ⇔ heure présente + ISO valide.
 * Retourne false si l'état UI serait incohérent (heure affichée mais ISO impossible).
 */
export const applyDepartureToPayload = (payload, { missionDate, pickupTime }) => {
  const pickupConfirmed = derivePickupTimeConfirmed(pickupTime);
  const pickupIso = combineMissionDateTime(missionDate, pickupTime);
  payload.pickup_time_confirmed = pickupConfirmed;
  if (pickupConfirmed) {
    if (!pickupIso) {
      return false;
    }
    payload.scheduled_time = pickupIso;
    payload.scheduled_time_type = 'departure';
  }
  return true;
};
