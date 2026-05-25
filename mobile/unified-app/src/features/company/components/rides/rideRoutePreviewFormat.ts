export const RIDE_PREVIEW_HEIGHT = 180;
export const RIDE_PREVIEW_EMPTY_HEIGHT = 96;

/** Format compact pour l'aperçu de course (`18 km`, `850 m`, ou `—`). */
export function formatRideDistance(meters: number | null | undefined): string {
  if (meters == null || !Number.isFinite(meters) || meters <= 0) return "—";
  if (meters < 1000) {
    return `${Math.round(meters)} m`;
  }
  const km = meters / 1000;
  if (km < 10) {
    return `${km.toFixed(1).replace(".", ",")} km`;
  }
  return `${Math.round(km)} km`;
}

/** Format compact pour l'aperçu de course (`24 min`, `1 h 12`, ou `—`). */
export function formatRideDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return "—";
  const totalMinutes = Math.max(1, Math.round(seconds / 60));
  if (totalMinutes < 60) return `${totalMinutes} min`;
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (minutes === 0) return `${hours} h`;
  return `${hours} h ${String(minutes).padStart(2, "0")}`;
}
