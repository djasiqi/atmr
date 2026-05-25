/** Durée max en mode reduce motion (fade minimal). */
export const REDUCE_MOTION_MAX_MS = 80;

/**
 * Retourne 0 (instantané) ou la durée demandée selon reduce motion.
 */
export function resolveMotionDuration(requestedMs: number, reduceMotion: boolean): number {
  if (reduceMotion) {
    return requestedMs <= 0 ? 0 : REDUCE_MOTION_MAX_MS;
  }
  return requestedMs;
}
