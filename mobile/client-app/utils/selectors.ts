import type { Booking } from '@/types/api';

function getBookingTime(b: Booking): number {
  const v = Date.parse(String(b.scheduled_time ?? ''));
  return Number.isFinite(v) ? v : 0;
}

export function selectNextBooking(bookings: Booking[]): Booking | null {
  const now = Date.now();
  const candidates = bookings
    .filter((b) => getBookingTime(b) >= now)
    .sort((a, b) => getBookingTime(a) - getBookingTime(b));
  return candidates[0] ?? null;
}
