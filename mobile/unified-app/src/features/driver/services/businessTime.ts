/**
 * Contrat temporel métier mobile (P0-F TIME) — Europe/Zurich.
 * Indépendant du timezone du téléphone.
 */

export const BUSINESS_TIME_ZONE = "Europe/Zurich";

/** Bornes présence GPS figées (P0-F TIME) — ne pas diverger du backend. */
export const PRESENCE_WINDOW_START_HOUR = 7;
export const PRESENCE_WINDOW_END_HOUR = 19;

export type BusinessTimeParts = {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
};

const PARTS_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: BUSINESS_TIME_ZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

export function getBusinessTimeParts(date: Date = new Date()): BusinessTimeParts {
  const parts = PARTS_FORMATTER.formatToParts(date);
  const get = (type: Intl.DateTimeFormatPartTypes): number => {
    const v = parts.find((p) => p.type === type)?.value;
    return v != null ? Number(v) : NaN;
  };
  return {
    year: get("year"),
    month: get("month"),
    day: get("day"),
    hour: get("hour"),
    minute: get("minute"),
    second: get("second"),
  };
}

/**
 * Instant UTC absolu correspondant à une horloge murale Europe/Zurich.
 * Gère le DST via sonde binaire (pas d’offset +1/+2 en dur).
 */
export function zonedWallClockToUtcDate(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute = 0,
  second = 0
): Date {
  // Estimation initiale : UTC ≈ murale (puis correction via parts Zurich)
  let guess = Date.UTC(year, month - 1, day, hour, minute, second);
  for (let i = 0; i < 4; i += 1) {
    const parts = getBusinessTimeParts(new Date(guess));
    const asUtc = Date.UTC(
      parts.year,
      parts.month - 1,
      parts.day,
      parts.hour,
      parts.minute,
      parts.second
    );
    const target = Date.UTC(year, month - 1, day, hour, minute, second);
    const delta = target - asUtc;
    if (delta === 0) break;
    guess += delta;
  }
  return new Date(guess);
}
