import dayjs from "dayjs";

type SchedulingLike = {
  time_defined?: boolean;
} | null | undefined;

type MissionLike = {
  scheduling?: SchedulingLike;
  time_confirmed?: boolean | null;
  scheduled_at?: string | null;
};

/** Heure non définie — préfère scheduling.time_defined / time_confirmed (INV-2). */
export function isTimeUndefined(mission: MissionLike | null | undefined): boolean {
  if (!mission) return true;
  const scheduling = mission.scheduling;
  if (scheduling && typeof scheduling.time_defined === "boolean") {
    return !scheduling.time_defined;
  }
  if (mission.time_confirmed === false) return true;
  return isPickupSentinel(mission.scheduled_at);
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
