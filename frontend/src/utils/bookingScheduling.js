import { renderBookingDateTime } from './formatDate';

/**
 * Formate l'horaire à partir du bloc API `scheduling` ou fallback legacy.
 */
export function formatAppointmentTime(booking, { dateAndTime = false } = {}) {
  const scheduling = booking?.scheduling;
  if (scheduling) {
    if (!scheduling.time_defined) {
      return 'À définir';
    }
    if (dateAndTime) {
      return renderBookingDateTime(booking);
    }
    return scheduling.display_time || renderBookingDateTime(booking);
  }

  if (booking?.time_confirmed === false) {
    return 'À définir';
  }
  if (!booking?.scheduled_time) {
    return 'À définir';
  }
  return renderBookingDateTime(booking);
}

export function isAppointmentTimeDefined(booking) {
  if (booking?.scheduling && typeof booking.scheduling.time_defined === 'boolean') {
    return booking.scheduling.time_defined;
  }
  if (booking?.time_confirmed === false) return false;
  return Boolean(booking?.scheduled_time);
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
