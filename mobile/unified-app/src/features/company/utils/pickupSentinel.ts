import dayjs from "dayjs";

type SchedulingLike = {
  time_defined?: boolean;
} | null | undefined;

type MissionLike = {
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

function resolveMissionTimeConfirmed(mission: unknown): boolean | null | undefined {
  const root = readNestedRecord(mission);
  const summary = readNestedRecord(root?.summary);
  const value = root?.time_confirmed ?? summary?.time_confirmed;
  return typeof value === "boolean" ? value : undefined;
}

/** Heure non définie — préfère scheduling.time_defined / time_confirmed (INV-2). */
export function isTimeUndefined(mission: MissionLike | null | undefined): boolean {
  if (!mission) return true;
  const scheduling = mission.scheduling ?? resolveMissionScheduling(mission);
  if (scheduling && typeof scheduling.time_defined === "boolean") {
    return !scheduling.time_defined;
  }
  const timeConfirmed = mission.time_confirmed ?? resolveMissionTimeConfirmed(mission);
  if (timeConfirmed === false) return true;
  return isPickupSentinel(resolveMissionScheduledAt(mission) ?? mission.scheduled_at);
}

/** Legacy : sentinelle T00:00:00 — conservé pour données sans time_confirmed. */
export function isPickupSentinel(pickupAt: string | null | undefined): boolean {
  if (pickupAt == null || pickupAt === "") return true;
  const d = dayjs(pickupAt);
  if (!d.isValid()) return true;
  const m = pickupAt.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === "00" && m[2] === "00" && m[3] === "00") return true;
  return false;
}

/**
 * Urgent autorisé uniquement tant que l’horaire n’est pas planifié (aligné POST …/urgent backend).
 * Une heure explicite (ex. 09:30) exclut l’urgence, même si `time_confirmed` est false (retour).
 */
export function canMarkRideUrgent(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  const scheduling = mission.scheduling ?? resolveMissionScheduling(mission);
  if (scheduling && scheduling.time_defined === true) return false;
  const timeConfirmed = mission.time_confirmed ?? resolveMissionTimeConfirmed(mission);
  if (timeConfirmed === true) return false;
  return isPickupSentinel(resolveMissionScheduledAt(mission));
}

/** Course affichable dans la liste dispatch (sans horaire, « à définir » ou date valide). */
export function missionHasRenderableSchedule(mission: MissionLike | null | undefined): boolean {
  if (!mission) return false;
  const at = resolveMissionScheduledAt(mission) ?? mission.scheduled_at;
  if (at == null || at === "") return true;
  if (isTimeUndefined(mission)) return true;
  return Number.isFinite(Date.parse(at));
}
