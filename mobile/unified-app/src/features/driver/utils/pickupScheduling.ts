import type { DriverMission } from "../types";

type SchedulingLike = {
  time_defined?: boolean;
  time_scheduled?: boolean;
} | null | undefined;

function isPickupSentinelLegacy(
  scheduledTime: string | null | undefined,
  timeConfirmed?: boolean | null
): boolean {
  if (scheduledTime == null || scheduledTime === "") return true;
  const raw = String(scheduledTime).trim();
  const m = raw.match(/T(\d{2}):(\d{2}):(\d{2})/);
  if (m && m[1] === "00" && m[2] === "00" && m[3] === "00") {
    if (timeConfirmed === true) return false;
    return true;
  }
  return false;
}

function resolveScheduling(trip: DriverMission | null | undefined): SchedulingLike {
  const scheduling = trip?.scheduling;
  return scheduling && typeof scheduling === "object" ? (scheduling as SchedulingLike) : null;
}

function resolveTimeConfirmed(trip: DriverMission | null | undefined): boolean | null | undefined {
  const value = trip?.time_confirmed;
  return typeof value === "boolean" ? value : undefined;
}

/** Existence d'une heure métier (tri, affichage « Heure à définir »). */
export function driverHasScheduledPickupTime(trip: DriverMission | null | undefined): boolean {
  if (!trip) return false;
  const scheduling = resolveScheduling(trip);
  if (scheduling && typeof scheduling.time_scheduled === "boolean") {
    return scheduling.time_scheduled;
  }
  const at = trip.scheduled_time;
  if (at == null || at === "") return false;
  return !isPickupSentinelLegacy(at, resolveTimeConfirmed(trip));
}

/** Heure confirmée workflow INV-2 (retards). */
export function driverHasConfirmedPickupTime(trip: DriverMission | null | undefined): boolean {
  if (!trip) return false;
  const scheduling = resolveScheduling(trip);
  if (scheduling && typeof scheduling.time_defined === "boolean") {
    return scheduling.time_defined;
  }
  const timeConfirmed = resolveTimeConfirmed(trip);
  if (timeConfirmed === true) return driverHasScheduledPickupTime(trip);
  if (timeConfirmed === false) return false;
  return driverHasScheduledPickupTime(trip);
}

export function formatDriverScheduleTimeLabel(mission: DriverMission | null | undefined): string {
  if (!driverHasScheduledPickupTime(mission)) return "Heure à définir";
  const raw = mission?.scheduled_time;
  const ts = Date.parse(String(raw ?? ""));
  if (!Number.isFinite(ts)) return "Heure à définir";
  const time = new Date(ts).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  if (resolveTimeConfirmed(mission) === false) return `${time} (non confirmé)`;
  return time;
}
