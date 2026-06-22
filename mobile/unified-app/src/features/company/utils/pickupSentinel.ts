import dayjs from "dayjs";

type SchedulingLike = {
  time_defined?: boolean;
  time_scheduled?: boolean;
} | null | undefined;

export type MissionLike = {
  scheduling?: SchedulingLike;
  time_confirmed?: boolean | null;
  scheduled_at?: string | null;
  scheduled_time?: string | null;
  pickup_at?: string | null;
  time?: { pickup_at?: string | null } | null;
  summary?: Record<string, unknown> | null;
};

function readNestedRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Résout l’ISO horaire depuis les payloads liste, détail (`summary.time.pickup_at`), etc. */
export function resolveMissionScheduledAt(mission: unknown): string | null {
  const queue: Record<string, unknown>[] = [];
  const seen = new Set<unknown>();
  const root = readNestedRecord(mission);
  if (root) queue.push(root);

  while (queue.length > 0) {
    const raw = queue.shift();
    if (!raw || seen.has(raw)) continue;
    seen.add(raw);

    for (const key of ["scheduled_at", "scheduled_time", "pickup_at"] as const) {
      const value = raw[key];
      if (typeof value === "string" && value.trim()) return value.trim();
    }

    const time = readNestedRecord(raw.time);
    const pickupAt = time?.pickup_at;
    if (typeof pickupAt === "string" && pickupAt.trim()) return pickupAt.trim();

    for (const value of Object.values(raw)) {
      const nested = readNestedRecord(value);
      if (nested) queue.push(nested);
    }
  }
  return null;
}

function resolveMissionScheduling(mission: unknown): SchedulingLike {
  const root = readNestedRecord(mission);
  const summary = readNestedRecord(root?.summary);
  const scheduling = root?.scheduling ?? summary?.scheduling;
  return scheduling && typeof scheduling === "object" ? (scheduling as SchedulingLike) : null;
}

export function resolveMissionTimeConfirmed(mission: unknown): boolean | null | undefined {
  const root = readNestedRecord(mission);
  const summary = readNestedRecord(root?.summary);
  const value = root?.time_confirmed ?? summary?.time_confirmed;
  return typeof value === "boolean" ? value : undefined;
}

/** Legacy : sentinelle T00:00:00 — fallback client si `scheduling.time_scheduled` absent. */
export function isPickupSentinel(
  pickupAt: string | null | undefined,
  timeConfirmed?: boolean | null
): boolean {
  if (pickupAt == null || pickupAt === "") return true;
  const d = dayjs(pickupAt);
  if (!d.isValid()) return true;
  const m = pickupAt.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === "00" && m[2] === "00" && m[3] === "00") {
    if (timeConfirmed === true) return false;
    return true;
  }
  return false;
}

/** Existence d'une heure métier (urgence, « À définir », tri sans heure). */
export function hasScheduledPickupTime(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  const scheduling = mission.scheduling ?? resolveMissionScheduling(mission);
  if (scheduling && typeof scheduling.time_scheduled === "boolean") {
    return scheduling.time_scheduled;
  }
  const at = resolveMissionScheduledAt(mission) ?? mission.scheduled_at;
  if (at == null || at === "") return false;
  const timeConfirmed = mission.time_confirmed ?? resolveMissionTimeConfirmed(mission);
  return !isPickupSentinel(at, timeConfirmed);
}

/** Heure confirmée workflow INV-2 (retards, dispatch opérationnel). */
export function hasConfirmedPickupTime(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  const scheduling = mission.scheduling ?? resolveMissionScheduling(mission);
  if (scheduling && typeof scheduling.time_defined === "boolean") {
    return scheduling.time_defined;
  }
  const timeConfirmed = mission.time_confirmed ?? resolveMissionTimeConfirmed(mission);
  if (timeConfirmed === true) return hasScheduledPickupTime(mission);
  if (timeConfirmed === false) return false;
  return hasScheduledPickupTime(mission);
}

export function isTimeUndefined(mission: MissionLike | null | undefined): boolean {
  return !hasScheduledPickupTime(mission);
}

/** Urgent autorisé uniquement sans heure métier (aligné POST …/urgent backend). */
export function canMarkRideUrgent(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  return !hasScheduledPickupTime(mission);
}

/** Course affichable dans la liste dispatch (sans horaire, « à définir » ou date valide). */
export function missionHasRenderableSchedule(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  const at = resolveMissionScheduledAt(mission) ?? mission.scheduled_at;
  if (at == null || at === "") return true;
  if (isTimeUndefined(mission)) return true;
  return Number.isFinite(Date.parse(at));
}
