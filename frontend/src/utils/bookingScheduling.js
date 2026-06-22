import { renderBookingDateTime } from './formatDate';

/** Legacy : sentinelle T00:00:00 — fallback si scheduling.time_scheduled absent. */
export function isPickupSentinel(scheduledTime, timeConfirmed) {
  if (scheduledTime == null || scheduledTime === '') return true;
  const raw = String(scheduledTime).trim();
  const m = raw.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === '00' && m[2] === '00' && m[3] === '00') {
    if (timeConfirmed === true) return false;
    return true;
  }
  return false;
}

/** Existence d'une heure métier (urgence, « À définir », tri sans heure). */
export function hasScheduledPickupTime(booking) {
  if (!booking) return false;
  if (booking.scheduling && typeof booking.scheduling.time_scheduled === 'boolean') {
    return booking.scheduling.time_scheduled;
  }
  if (!booking.scheduled_time) return false;
  return !isPickupSentinel(booking.scheduled_time, booking.time_confirmed);
}

/** Heure confirmée workflow INV-2 (retards, dispatch opérationnel). */
export function hasConfirmedPickupTime(booking) {
  if (!booking) return false;
  if (booking.scheduling && typeof booking.scheduling.time_defined === 'boolean') {
    return booking.scheduling.time_defined;
  }
  if (booking.time_confirmed === true) return hasScheduledPickupTime(booking);
  if (booking.time_confirmed === false) return false;
  return hasScheduledPickupTime(booking);
}

/**
 * Formate l'horaire à partir du bloc API `scheduling` ou fallback legacy.
 */
export function formatAppointmentTime(booking, { dateAndTime = false } = {}) {
  const scheduling = booking?.scheduling;
  if (scheduling?.display_time) {
    if (!hasScheduledPickupTime(booking)) {
      return 'À définir';
    }
    if (dateAndTime) {
      return scheduling.display_datetime || renderBookingDateTime(booking);
    }
    return scheduling.display_time;
  }

  if (!hasScheduledPickupTime(booking)) {
    return 'À définir';
  }
  if (dateAndTime) {
    return renderBookingDateTime(booking);
  }
  return renderBookingDateTime(booking);
}

export function isAppointmentTimeDefined(booking) {
  return hasConfirmedPickupTime(booking);
}

/** Retour ou leg multi-étapes institution sans horaire opérationnel — pas d'assignation chauffeur. */
export function needsTimeBeforeDriverAssign(booking) {
  if (isAppointmentTimeDefined(booking)) return false;
  const status = String(booking?.status ?? '').toLowerCase();
  const isReturn = !!(
    booking?.is_return ||
    booking?.booking_type === 'return' ||
    booking?.type === 'return'
  );
  if (isReturn) return true;
  if (booking?.route_group_id && ['accepted', 'assigned'].includes(status)) {
    return true;
  }
  return false;
}

export function isReturnLegNeedingTime(booking) {
  const isReturn = !!(
    booking?.is_return ||
    booking?.booking_type === 'return' ||
    booking?.type === 'return'
  );
  return isReturn && !isAppointmentTimeDefined(booking);
}
