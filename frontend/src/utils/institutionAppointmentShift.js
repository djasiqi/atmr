/**
 * Décalage de rendez-vous institution — libellés pour reconfirmation transporteur.
 */

import { extractWallClockTime } from './missionTimeDisplay';

const HHMM_RE = /^(\d{1,2}):(\d{2})$/;

/** Normalise une heure murale en HH:MM (chaîne vide si invalide). */
export function normalizeHhmm(value) {
  if (value == null) return '';
  const raw = String(value).trim();
  if (!raw) return '';
  const hm = raw.match(HHMM_RE);
  if (hm) {
    const hour = Number(hm[1]);
    const minute = Number(hm[2]);
    if (hour > 23 || minute > 59) return '';
    return `${String(hour).padStart(2, '0')}:${hm[2]}`;
  }
  return extractWallClockTime(raw) || '';
}

function readSearchParam(searchParams, key) {
  if (!searchParams) return '';
  if (typeof searchParams.get === 'function') {
    return searchParams.get(key) || '';
  }
  return searchParams[key] || '';
}

function latestAppointmentEvent(events) {
  if (!Array.isArray(events) || events.length === 0) return null;
  return [...events]
    .sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0))
    .find((ev) => {
      const after = ev?.after_snapshot || {};
      const before = ev?.before_snapshot || {};
      return Boolean(
        after.appointment_before
        || after.appointment_after
        || after.appointment_time
        || before.appointment_time,
      );
    }) || null;
}

/**
 * Résout l'ancien et le nouveau RDV (HH:MM).
 * Priorité : query notif → dernier change-event → RDV actuel du booking.
 */
export function resolveAppointmentShift({
  reservation,
  events = [],
  searchParams,
} = {}) {
  let before = normalizeHhmm(
    readSearchParam(searchParams, 'appt_from')
    || readSearchParam(searchParams, 'appointment_before'),
  );
  let after = normalizeHhmm(
    readSearchParam(searchParams, 'appt_to')
    || readSearchParam(searchParams, 'appointment_after'),
  );

  if (!before || !after) {
    const ev = latestAppointmentEvent(events);
    if (ev) {
      const snap = ev.after_snapshot || {};
      const beforeSnap = ev.before_snapshot || {};
      before = before || normalizeHhmm(snap.appointment_before)
        || normalizeHhmm(beforeSnap.appointment_time);
      after = after || normalizeHhmm(snap.appointment_after)
        || normalizeHhmm(snap.appointment_time);
    }
  }

  if (!after) {
    after = normalizeHhmm(reservation?.institution_leg?.appointment_time);
  }

  return { before, after };
}

export function formatAppointmentShiftLead({ before, after } = {}) {
  if (before && after && before !== after) {
    return `Le rendez-vous a été décalé de ${before} à ${after}.`;
  }
  if (after) {
    return `Le rendez-vous a été décalé à ${after}.`;
  }
  return 'Le rendez-vous a été décalé.';
}
