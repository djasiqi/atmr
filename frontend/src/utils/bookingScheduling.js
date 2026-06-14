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
