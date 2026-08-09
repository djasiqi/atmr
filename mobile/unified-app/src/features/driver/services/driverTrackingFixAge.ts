/** Âge max d'un fix watch avant fallback getCurrentPosition (tous modes). */
export const WATCH_STALE_MS = 25_000;

/** Âge effectif du fix = max(âge timestamp GPS, âge lastWatchAt). */
export function computeFixAgeMs(
  position: { timestamp?: number | null },
  watchAtMs: number | null,
  nowMs: number = Date.now()
): number {
  const fromTimestamp =
    typeof position.timestamp === "number" && Number.isFinite(position.timestamp)
      ? nowMs - position.timestamp
      : 0;
  const fromWatch = watchAtMs != null ? nowMs - watchAtMs : 0;
  return Math.max(fromTimestamp, fromWatch);
}
