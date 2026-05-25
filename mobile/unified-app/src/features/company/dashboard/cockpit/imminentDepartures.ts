import type { CompanyDispatchMission } from "../../api/contracts";
import { isMissionDelayed, toEpoch } from "../companyDashboardMissionUi";

const IMMINENT_WINDOW_MS = 60 * 60 * 1000;
const INDIVIDUAL_THRESHOLD_MS = 15 * 60 * 1000;
/** Halo pickup conservé après l'heure prévue tant que la mission est active et en retard. */
const DELAYED_PICKUP_HIGHLIGHT_MS = 4 * 60 * 60 * 1000;

export type ImminentDepartureRisk = "normal" | "warning" | "critical";

export type ImminentDeparture = {
  missionId: number;
  scheduledAtMs: number;
  minutesUntil: number;
  risk: ImminentDepartureRisk;
  clusterKey: string | null;
  pickupLat: number | null;
  pickupLon: number | null;
};

export type ImminentDeparturesResult = {
  individual: ImminentDeparture[];
  clustered: ImminentDeparture[];
};

function computeRisk(
  mission: CompanyDispatchMission,
  minutesUntil: number,
  hasNearbyDriver: boolean,
  nowMs: number
): ImminentDepartureRisk {
  const delayMin = Number(mission.assignment_pickup_delay_minutes);
  const hasAssignmentDelay = Number.isFinite(delayMin) && delayMin > 0;
  if (isMissionDelayed(mission, nowMs) || hasAssignmentDelay) {
    return "critical";
  }
  const unassigned = mission.driver_id == null;
  if (unassigned && minutesUntil < 20) return "critical";
  if (!hasNearbyDriver && minutesUntil < 25) return "warning";
  return "normal";
}

function isActiveMission(status: CompanyDispatchMission["status"]): boolean {
  return status !== "completed" && status !== "cancelled";
}

function resolvePickupHighlight(
  mission: CompanyDispatchMission,
  nowMs: number
): { scheduledAtMs: number; minutesUntil: number } | null {
  if (!isActiveMission(mission.status)) return null;
  if (mission.pickup_lat == null || mission.pickup_lon == null) return null;

  const delayMin = Number(mission.assignment_pickup_delay_minutes);
  const hasAssignmentDelay = Number.isFinite(delayMin) && delayMin > 0;
  const delayed = isMissionDelayed(mission, nowMs) || hasAssignmentDelay;

  const scheduledAtMs = toEpoch(mission.scheduled_at);
  if (scheduledAtMs == null) {
    if (!delayed) return null;
    return { scheduledAtMs: nowMs, minutesUntil: 0 };
  }

  const delta = scheduledAtMs - nowMs;
  if (delta >= 0 && delta <= IMMINENT_WINDOW_MS) {
    return { scheduledAtMs, minutesUntil: Math.round(delta / 60_000) };
  }
  if (delta < 0 && delayed && nowMs - scheduledAtMs <= DELAYED_PICKUP_HIGHLIGHT_MS) {
    return { scheduledAtMs, minutesUntil: Math.round(delta / 60_000) };
  }
  return null;
}

export function buildImminentDepartures(
  missions: CompanyDispatchMission[],
  nowMs = Date.now(),
  hasDriverNearPickup?: (mission: CompanyDispatchMission) => boolean
): ImminentDeparturesResult {
  const individual: ImminentDeparture[] = [];
  const clustered: ImminentDeparture[] = [];

  for (const mission of missions) {
    const highlight = resolvePickupHighlight(mission, nowMs);
    if (!highlight) continue;

    const { scheduledAtMs, minutesUntil } = highlight;
    const delta = scheduledAtMs - nowMs;
    const nearby = hasDriverNearPickup?.(mission) ?? mission.driver_id != null;
    const item: ImminentDeparture = {
      missionId: mission.mission_id,
      scheduledAtMs,
      minutesUntil,
      risk: computeRisk(mission, minutesUntil, nearby, nowMs),
      clusterKey: minutesUntil >= 15 ? `bucket-${Math.floor(minutesUntil / 15)}` : null,
      pickupLat: mission.pickup_lat ?? null,
      pickupLon: mission.pickup_lon ?? null,
    };

    if (delta < INDIVIDUAL_THRESHOLD_MS) {
      individual.push(item);
    } else {
      clustered.push(item);
    }
  }

  individual.sort((a, b) => a.scheduledAtMs - b.scheduledAtMs);
  clustered.sort((a, b) => a.scheduledAtMs - b.scheduledAtMs);

  return { individual, clustered };
}
