/**
 * Bannière dashboard — chauffeurs ASSIGNED avec batterie restreinte et mission imminente.
 */

import { isDriverConstrained } from './companyDriverProjections';

export const CONSTRAINED_IMMINENT_TOAST_ID = 'dashboard-constrained-imminent';
export const CONSTRAINED_IMMINENT_WINDOW_MS = 30 * 60 * 1000;

function parseScheduledTimeMs(value) {
  if (value == null || value === '') return null;
  const t = Date.parse(String(value));
  return Number.isFinite(t) ? t : null;
}

function isAssignedDriverStatus(driver) {
  const status = String(driver?.status || '').toLowerCase();
  return status === 'assigned' || status === 'assigned_constrained';
}

/**
 * Chauffeurs assignés avec batterie restreinte dont la mission démarre dans < windowMs.
 * @param {Array} drivers
 * @param {Array} reservations
 * @param {number} nowMs
 * @param {number} [windowMs]
 */
export function countConstrainedAssignedImminentDrivers(
  drivers,
  reservations,
  nowMs,
  windowMs = CONSTRAINED_IMMINENT_WINDOW_MS
) {
  if (!Array.isArray(drivers) || drivers.length === 0) return 0;

  const bookingTimeByDriverId = new Map();
  if (Array.isArray(reservations)) {
    reservations.forEach((r) => {
      const driverId = r?.driver_id ?? r?.driver?.id;
      if (driverId == null) return;
      const scheduledMs = parseScheduledTimeMs(r.scheduled_time ?? r.pickup_time);
      if (scheduledMs == null) return;
      const normalizedId = Number(driverId);
      if (!Number.isFinite(normalizedId)) return;
      const prev = bookingTimeByDriverId.get(normalizedId);
      if (prev == null || scheduledMs < prev) {
        bookingTimeByDriverId.set(normalizedId, scheduledMs);
      }
    });
  }

  let count = 0;
  drivers.forEach((driver) => {
    if (!isAssignedDriverStatus(driver)) return;
    if (!isDriverConstrained(driver)) return;

    const scheduledMs = bookingTimeByDriverId.get(Number(driver.id));
    if (scheduledMs == null) return;

    const deltaMs = scheduledMs - nowMs;
    if (deltaMs < 0 || deltaMs > windowMs) return;
    count += 1;
  });

  return count;
}

/**
 * Message toast pour la bannière imminente.
 * @param {number} count
 */
export function buildConstrainedImminentToastMessage(count) {
  const n = Number(count) || 0;
  if (n <= 0) return '';
  const label = n === 1 ? 'chauffeur ASSIGNED a' : 'chauffeurs ASSIGNED ont';
  return `Attention: ${n} ${label} un problème d'optimisation batterie alors qu'une mission est imminente. Contactez-les avant l'heure prévue.`;
}
