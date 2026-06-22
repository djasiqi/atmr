import { isTimeUndefined } from "./pickupSentinel";

const SWISS_TZ = "Europe/Zurich";

const NAIVE_ISO_RE = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/;

/** Horodatage Genève naïf (`YYYY-MM-DDTHH:mm:ss`) à partir d’un instant. */
export function formatNaiveIsoInZurich(d: Date): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: SWISS_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(d);
  const get = (type: Intl.DateTimeFormatPartTypes) =>
    parts.find((p) => p.type === type)?.value ?? "00";
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}:${get("second")}`;
}

/**
 * Parse une heure planifiée API ou formulaire.
 * - ISO avec `Z` ou offset → instant UTC correct (comme `formatMissionTime`).
 * - ISO naïf → heure murale Genève (convention backend).
 */
export function parseScheduledTimeInstant(raw: string | null | undefined): Date | null {
  if (raw == null) return null;
  const t = raw.trim();
  if (!t) return null;

  if (/[zZ]$/.test(t) || /(?:[+-]\d{2}:\d{2})$/.test(t)) {
    const d = new Date(t);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const base = t.replace(/\.\d+$/, "").replace(/Z$/i, "");
  const m = NAIVE_ISO_RE.exec(base);
  if (!m) {
    const d = new Date(base.includes("T") ? base : `${base}T00:00:00`);
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const wall = `${m[1]}T${m[2]}:${m[3]}:${m[4] ?? "00"}`;
  for (const offset of ["+02:00", "+01:00"]) {
    const d = new Date(`${wall}${offset}`);
    if (Number.isNaN(d.getTime())) continue;
    if (formatNaiveIsoInZurich(d) === wall) return d;
  }

  const fallback = new Date(`${wall}+01:00`);
  return Number.isNaN(fallback.getTime()) ? null : fallback;
}

/** Convertit l’ISO API (souvent UTC « Z ») en ISO naïf Genève pour les formulaires. */
export function scheduledTimeToFormNaiveIso(raw: string | null | undefined): string {
  if (raw == null) return "";
  const t = raw.trim();
  if (!t) return "";

  if (!/[zZ]$/.test(t) && !/(?:[+-]\d{2}:\d{2})$/.test(t)) {
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(t)) {
      return t.includes(".") ? t.split(".")[0] : t;
    }
    const short = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})$/.exec(t);
    if (short) return `${short[1]}:00`;
    return t;
  }

  const instant = parseScheduledTimeInstant(t);
  if (!instant) return t.replace(/Z$/i, "");
  return formatNaiveIsoInZurich(instant);
}

type ZurichWallParts = {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
};

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

export function zurichWallPartsFromDate(d: Date): ZurichWallParts {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: SWISS_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(d);
  const num = (type: Intl.DateTimeFormatPartTypes) =>
    Number(parts.find((p) => p.type === type)?.value ?? 0);
  return {
    year: num("year"),
    month: num("month"),
    day: num("day"),
    hour: num("hour"),
    minute: num("minute"),
    second: num("second"),
  };
}

export function dateFromZurichWallParts(
  year: number,
  month: number,
  day: number,
  hour = 0,
  minute = 0,
  second = 0,
): Date {
  const wall = `${year}-${pad2(month)}-${pad2(day)}T${pad2(hour)}:${pad2(minute)}:${pad2(second)}`;
  const parsed = parseScheduledTimeInstant(wall);
  if (!parsed) {
    throw new Error(`Date Genève invalide : ${wall}`);
  }
  return parsed;
}

export function startOfZurichDay(d: Date): Date {
  const p = zurichWallPartsFromDate(d);
  return dateFromZurichWallParts(p.year, p.month, p.day, 0, 0, 0);
}

export function isSameZurichDay(a: Date, b: Date): boolean {
  const pa = zurichWallPartsFromDate(a);
  const pb = zurichWallPartsFromDate(b);
  return pa.year === pb.year && pa.month === pb.month && pa.day === pb.day;
}

/** Conserve l’heure Genève de `timeSource` sur le jour calendaire Genève de `daySource`. */
export function mergeZurichDayAndTime(daySource: Date, timeSource: Date): Date {
  const day = zurichWallPartsFromDate(daySource);
  const time = zurichWallPartsFromDate(timeSource);
  return dateFromZurichWallParts(day.year, day.month, day.day, time.hour, time.minute, 0);
}

export function getTodayStartInZurich(): Date {
  return startOfZurichDay(new Date());
}

export function clampZurichDayToToday(day: Date, todayStartZurich: Date): Date {
  const picked = startOfZurichDay(day);
  const today = startOfZurichDay(todayStartZurich);
  return picked.getTime() < today.getTime() ? new Date(today) : picked;
}

/** Jour affiché dans la bande (calendrier local) → horaire Genève avec l’heure de `timeSource`. */
export function buildGenevaScheduleFromLocalCalendarDay(
  localCalendarDay: Date,
  timeSource: Date,
): Date {
  const time = zurichWallPartsFromDate(timeSource);
  let year = localCalendarDay.getFullYear();
  let month = localCalendarDay.getMonth() + 1;
  let day = localCalendarDay.getDate();
  const noon = dateFromZurichWallParts(year, month, day, 12, 0, 0);
  const todayStart = startOfZurichDay(new Date());
  if (startOfZurichDay(noon).getTime() < todayStart.getTime()) {
    const t = zurichWallPartsFromDate(todayStart);
    year = t.year;
    month = t.month;
    day = t.day;
  }
  return dateFromZurichWallParts(year, month, day, time.hour, time.minute, 0);
}

export function isFutureZurichDay(candidate: Date, todayStartZurich: Date): boolean {
  return startOfZurichDay(candidate).getTime() > startOfZurichDay(todayStartZurich).getTime();
}

type MissionDateLike = {
  scheduled_at?: string | null;
  scheduling?: { time_defined?: boolean } | null;
  time_confirmed?: boolean | null;
};

function zonedIsoDate(epochMs: number): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: SWISS_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(epochMs));
  const year = parts.find((p) => p.type === "year")?.value;
  const month = parts.find((p) => p.type === "month")?.value;
  const day = parts.find((p) => p.type === "day")?.value;
  return `${year}-${month}-${day}`;
}

export function isoDateInZurichFromIso(iso: string | null | undefined): string | null {
  if (iso == null || iso === "") return null;
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return null;
  return zonedIsoDate(ms);
}

/** Date du jour en Europe/Zurich (aligné backend day_local_bounds). */
export function getTodayIsoDateInZurich(): string {
  return zonedIsoDate(Date.now());
}

/**
 * Course visible pour le jour sélectionné.
 * Exclut les missions datées explicitement sur un autre jour (ex. heure non définie d'un autre jour).
 * Conserve les retours sans horaire renvoyés par l'API pour l'aller du jour.
 */
export function missionBelongsToSelectedDay(
  mission: MissionDateLike | null | undefined,
  selectedDateIso: string
): boolean {
  if (!mission) return false;
  const at = mission.scheduled_at;
  if (at == null || at === "") return true;
  const missionDate = isoDateInZurichFromIso(at);
  if (missionDate == null) return true;
  if (isTimeUndefined(mission)) {
    return missionDate === selectedDateIso;
  }
  return missionDate === selectedDateIso;
}
