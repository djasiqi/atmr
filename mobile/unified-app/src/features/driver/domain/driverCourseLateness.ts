import type { DriverMission, DriverMissionStatus } from "../types";
import { getDriverStatusUx, normalizeDriverMissionStatus } from "../statusDictionary";
import {
  driverHasConfirmedPickupTime,
  driverHasScheduledPickupTime,
} from "../utils/pickupScheduling";

/**
 * Retard de prise en charge déjà utilisé par la liste chauffeur :
 * heure confirmée dépassée, sans nouveau calcul.
 * Le rouge s'arrête dès que ce retard n'est plus le statut opérationnel
 * (arrivée, patient à bord, course terminée ou annulée).
 */
const PICKUP_DELAY_STATUSES = new Set<DriverMissionStatus>(["ASSIGNED", "EN_ROUTE"]);

export function driverCoursePickupIsLate(mission: DriverMission, nowTs: number): boolean {
  const key = normalizeDriverMissionStatus(mission.status);
  if (!PICKUP_DELAY_STATUSES.has(key)) return false;
  if (!driverHasConfirmedPickupTime(mission)) return false;
  if (!driverHasScheduledPickupTime(mission)) return false;
  const raw = mission.scheduled_time;
  if (raw == null || !String(raw).trim()) return false;
  const ts = Date.parse(String(raw));
  if (!Number.isFinite(ts)) return false;
  return ts < nowTs;
}

export function driverCourseStatusLabel(status: string | null | undefined): string {
  return getDriverStatusUx(status).label;
}

export type DriverCourseBadgeTone = {
  background: string;
  foreground: string;
  icon:
    | "time-outline"
    | "navigate-outline"
    | "location-outline"
    | "person-outline"
    | "checkmark-circle-outline"
    | "close-circle-outline"
    | "ellipse-outline";
};

const LATE_TONE = {
  background: "#FEE2E2",
  foreground: "#991B1B",
} as const;

export function driverCourseBadgeTone(
  status: string | null | undefined,
  late: boolean
): DriverCourseBadgeTone {
  const key = normalizeDriverMissionStatus(status);
  const icon = iconForStatus(key);
  if (late && PICKUP_DELAY_STATUSES.has(key)) {
    return { ...LATE_TONE, icon };
  }
  switch (key) {
    case "ASSIGNED":
      return { background: "#F1F5F9", foreground: "#475569", icon };
    case "EN_ROUTE":
      return { background: "#DBEAFE", foreground: "#1D4ED8", icon };
    case "ARRIVED":
      return { background: "#CCFBF1", foreground: "#0F766E", icon };
    case "IN_PROGRESS":
      return { background: "#DCFCE7", foreground: "#15803D", icon };
    case "COMPLETED":
      return { background: "#F1F5F9", foreground: "#64748B", icon };
    case "CANCELLED":
    case "REASSIGNED":
    case "NO_SHOW":
    case "EXPIRED":
    case "FAILED":
      return { background: "#F1F5F9", foreground: "#64748B", icon };
    default:
      return { background: "#F1F5F9", foreground: "#475569", icon };
  }
}

function iconForStatus(key: DriverMissionStatus): DriverCourseBadgeTone["icon"] {
  switch (key) {
    case "ASSIGNED":
      return "time-outline";
    case "EN_ROUTE":
      return "navigate-outline";
    case "ARRIVED":
      return "location-outline";
    case "IN_PROGRESS":
      return "person-outline";
    case "COMPLETED":
      return "checkmark-circle-outline";
    case "CANCELLED":
    case "REASSIGNED":
    case "NO_SHOW":
    case "EXPIRED":
    case "FAILED":
      return "close-circle-outline";
    default:
      return "ellipse-outline";
  }
}
