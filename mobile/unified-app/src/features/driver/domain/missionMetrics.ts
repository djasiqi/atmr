import type { DriverMission } from "../types";

export {
  compactMissionDistance,
  compactMissionEta,
  formatMissionRouteDistanceKm,
  formatMissionRouteDurationMinutes,
  resolveMissionRouteMetrics,
} from "./missionRouteMetrics";

const SWISS_TZ = "Europe/Zurich";

/** Heure de prise en charge — ex. `10:50`. */
export function formatMissionPickupTime(mission: DriverMission): string {
  const raw =
    typeof mission.scheduled_time === "string" && mission.scheduled_time.length > 0
      ? mission.scheduled_time
      : typeof mission.scheduled_at === "string"
        ? (mission.scheduled_at as string)
        : null;
  if (!raw) return "—";
  const d = new Date(raw);
  if (!Number.isFinite(d.getTime())) return "—";
  return d.toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
}
