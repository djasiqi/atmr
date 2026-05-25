/** Timings centralisés — Operational Calm (pas de durées hardcodées dans les composants). */
export const COCKPIT_MOTION_TOKENS = {
  FAST_FADE_MS: 120,
  STANDARD_FADE_MS: 220,
  CONTEXT_FADE_MS: 350,
  CAMERA_SOFT_MS: 600,
  ROUTE_ENTER_MS: 380,
  ROUTE_EXIT_MS: 420,
  INCIDENT_PULSE_MS: 800,
  SELECTION_HIGHLIGHT_MS: 200,
} as const;

export type CockpitMotionTokenKey = keyof typeof COCKPIT_MOTION_TOKENS;
