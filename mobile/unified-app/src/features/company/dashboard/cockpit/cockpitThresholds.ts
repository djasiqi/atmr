/** Seuils cockpit centralisés (Phase 4 — dette technique). */

export const COCKPIT_HEALTH_SAFE_THRESHOLD = 35;
export const COCKPIT_CHAOS_URGENT_THRESHOLD = 3;
export const COCKPIT_CHAOS_DELAYED_THRESHOLD = 4;

export const DENSITY_DRIVER_THRESHOLDS = {
  medium: 15,
  high: 30,
  extreme: 50,
  aggregate: 100,
} as const;

export const COCKPIT_EVENT_BUS_TTL_MS = 5 * 60_000;
