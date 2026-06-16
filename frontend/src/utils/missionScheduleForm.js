/**
 * Helpers formulaire institution — date mission + heures par étape.
 * Règle : heure saisie ⇒ confirmée ; heure vide ⇒ non confirmée.
 */

import {
  combineMissionDateTimeNaive,
  extractWallClockTime,
  minutesSinceMissionWallClock,
} from './missionTimeDisplay';

export const pad2 = (n) => String(n).padStart(2, '0');

/** Date locale YYYY-MM-DD (raccourcis formulaire, fuseau navigateur). */
export const formatLocalDateYMD = (date) =>
  `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;

/** Heure locale HH:MM (raccourcis formulaire). */
export const formatLocalTimeHM = (date) => `${pad2(date.getHours())}:${pad2(date.getMinutes())}`;

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
  return combineMissionDateTimeNaive(normalizedDate, timeHHMM?.trim());
};

/** Délai minimal (minutes) entre maintenant et un rendez-vous / arrivée. */
export const MIN_ARRIVAL_LEAD_MINUTES = 60;

/** Tolérance (minutes) pour le départ afin d'accepter « maintenant » (raccourci Urgent). */
const PICKUP_PAST_GRACE_MINUTES = 2;

/** True si l'instant ISO est dans le passé (au-delà de la tolérance). */
export const isInstantInPast = (iso, graceMinutes = PICKUP_PAST_GRACE_MINUTES) => {
  if (!iso) return false;
  const elapsed = minutesSinceMissionWallClock(iso);
  if (!Number.isFinite(elapsed)) return false;
  return elapsed > graceMinutes;
};

/** True si l'instant ISO est avant maintenant + leadMinutes (rendez-vous trop proche/passé). */
export const isInstantBeforeLead = (iso, leadMinutes = MIN_ARRIVAL_LEAD_MINUTES) => {
  if (!iso) return false;
  const elapsed = minutesSinceMissionWallClock(iso);
  if (!Number.isFinite(elapsed)) return false;
  return elapsed > -leadMinutes;
};

/** Normalise une valeur d'heure (ISO, "YYYY-MM-DDTHH:MM" ou "HH:MM") en "HH:MM". */
export const extractHHMM = (value) => {
  if (!value) return '';
  const raw = String(value).trim();
  if (/^\d{2}:\d{2}$/.test(raw)) return raw;
  return extractWallClockTime(raw);
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
