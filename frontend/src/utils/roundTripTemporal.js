import { extractWallClockDate, extractWallClockTime } from './missionTimeDisplay';

export const RETURN_BEFORE_OUTBOUND_MESSAGE =
  "L'heure de retour ne peut pas être antérieure à l'aller.";

export const RETURN_BEFORE_APPOINTMENT_MESSAGE =
  "L'heure de retour ne peut pas précéder l'heure de rendez-vous.";

export function wallClockKey({ iso, date, time } = {}) {
  if (iso) {
    const ymd = extractWallClockDate(iso);
    const hhmm = extractWallClockTime(iso);
    if (ymd && hhmm) return `${ymd}T${hhmm}`;
  }
  const ymd = date ? String(date).trim() : '';
  const hhmm = time ? String(time).trim().slice(0, 5) : '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(ymd) && /^\d{2}:\d{2}$/.test(hhmm)) {
    return `${ymd}T${hhmm}`;
  }
  return null;
}

export function isReturnPickupPossible({
  outboundPickup,
  returnPickup,
  appointment,
} = {}) {
  if (!returnPickup) return true;
  if (outboundPickup && returnPickup <= outboundPickup) return false;
  if (appointment && returnPickup < appointment) return false;
  return true;
}

export function findLinkedOutbound(reservation, linkedBookings = []) {
  const rows = linkedBookings || [];
  if (reservation?.parent_booking_id != null) {
    const parentId = Number(reservation.parent_booking_id);
    const parent = rows.find((row) => Number(row.id) === parentId);
    if (parent) return parent;
  }
  const groupId = reservation?.route_group_id;
  const seq = Number(reservation?.route_sequence_number);
  if (groupId && Number.isFinite(seq) && seq > 1) {
    const previous = rows
      .filter((row) => (
        row.route_group_id === groupId
        && Number(row.route_sequence_number) < seq
      ))
      .sort((a, b) => Number(b.route_sequence_number) - Number(a.route_sequence_number));
    return previous[0] || null;
  }
  return null;
}

export function resolveReturnPickupConflict({
  reservation,
  linkedBookings = [],
  returnDate,
  returnTime,
  returnPickupIso,
  appointmentIso,
} = {}) {
  const outbound = findLinkedOutbound(reservation, linkedBookings);
  if (!outbound) return null;
  const outboundPickup = wallClockKey({
    iso: outbound?.scheduled_time || outbound?.scheduling?.scheduled_time,
  });
  const returnPickup = wallClockKey({
    iso: returnPickupIso,
    date: returnDate,
    time: returnTime,
  });
  const appointment = wallClockKey({ iso: appointmentIso });
  if (isReturnPickupPossible({ outboundPickup, returnPickup, appointment })) {
    return null;
  }
  if (appointment && returnPickup && returnPickup < appointment) {
    return RETURN_BEFORE_APPOINTMENT_MESSAGE;
  }
  return RETURN_BEFORE_OUTBOUND_MESSAGE;
}
