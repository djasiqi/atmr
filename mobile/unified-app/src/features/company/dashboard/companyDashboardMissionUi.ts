import type { CompanyDispatchMission, CompanyDispatchMissionStatus } from "../api/contracts";
import { isPickupSentinel, isTimeUndefined } from "../utils/pickupSentinel";

const SWISS_TZ = "Europe/Zurich";

export type MissionUiStatusTone = "assign" | "in_progress" | "upcoming" | "delayed" | "completed" | "cancelled";

export type MissionUiStatus = {
  label: string;
  tone: MissionUiStatusTone;
  barColor: string;
};

const TONE_COLORS: Record<MissionUiStatusTone, string> = {
  assign: "#F59E0B",
  in_progress: "#3B82F6",
  upcoming: "#94A3B8",
  delayed: "#EF4444",
  completed: "#64748B",
  cancelled: "#94A3B8",
};

export function toEpoch(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function formatMissionTime(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (!Number.isFinite(d.getTime())) return "—";
  return d.toLocaleTimeString("fr-CH", {
    timeZone: SWISS_TZ,
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Heure planifiée ou « À définir » si sentinel T00:00:00 (transport sans horaire fixé). */
export function formatMissionScheduleTimeLabel(value: string | null | undefined): string {
  if (isPickupSentinel(value)) return "À définir";
  return formatMissionTime(value);
}

export function formatMissionDateTimeShort(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (!Number.isFinite(d.getTime())) return "—";
  return d.toLocaleString("fr-CH", {
    timeZone: SWISS_TZ,
    weekday: "short",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function conciseRouteSegment(s: string | null | undefined, maxLen = 42): string {
  const t = s?.trim() ?? "";
  if (!t) return "—";
  const head = t.split(",")[0]?.trim() || t;
  if (head.length <= maxLen) return head;
  return `${head.slice(0, Math.max(0, maxLen - 1))}…`;
}

export function missionHasDefinedPickupTime(
  missionOrScheduledAt: CompanyDispatchMission | string | null | undefined
): boolean {
  if (missionOrScheduledAt && typeof missionOrScheduledAt === "object") {
    return !isTimeUndefined(missionOrScheduledAt);
  }
  return !isPickupSentinel(missionOrScheduledAt);
}

export function resolveMissionUiStatus(
  mission: CompanyDispatchMission,
  nowMs = Date.now()
): MissionUiStatus {
  const status = mission.status;
  const scheduleDefined = missionHasDefinedPickupTime(mission);
  const scheduled = scheduleDefined ? toEpoch(mission.scheduled_at) : 0;
  const delayMin = Number(mission.assignment_pickup_delay_minutes);
  const isDelayedByAssignment =
    scheduleDefined && Number.isFinite(delayMin) && delayMin > 0;
  const isPastScheduled =
    scheduleDefined &&
    scheduled > 0 &&
    scheduled < nowMs &&
    status !== "completed" &&
    status !== "cancelled";

  if (status === "completed") {
    return { label: "Terminée", tone: "completed", barColor: TONE_COLORS.completed };
  }
  if (status === "cancelled") {
    return { label: "Annulée", tone: "cancelled", barColor: TONE_COLORS.cancelled };
  }
  if (!scheduleDefined) {
    if (status === "en_route" || status === "in_progress") {
      return { label: "En cours", tone: "in_progress", barColor: TONE_COLORS.in_progress };
    }
    if (status === "pending" || status === "proposed" || status === "accepted") {
      return { label: "À assigner", tone: "assign", barColor: TONE_COLORS.assign };
    }
    if (status === "assigned") {
      return { label: "Heure à définir", tone: "assign", barColor: TONE_COLORS.assign };
    }
    return { label: "Heure à définir", tone: "upcoming", barColor: TONE_COLORS.upcoming };
  }
  if (isDelayedByAssignment || isPastScheduled) {
    return { label: "En retard", tone: "delayed", barColor: TONE_COLORS.delayed };
  }
  if (status === "en_route" || status === "in_progress") {
    return { label: "En cours", tone: "in_progress", barColor: TONE_COLORS.in_progress };
  }
  if (status === "pending" || status === "proposed" || status === "accepted") {
    return { label: "À assigner", tone: "assign", barColor: TONE_COLORS.assign };
  }
  if (status === "assigned") {
    return { label: "Assignée", tone: "upcoming", barColor: TONE_COLORS.upcoming };
  }
  return { label: "À venir", tone: "upcoming", barColor: TONE_COLORS.upcoming };
}

export function formatEtaLabel(mission: CompanyDispatchMission): string | null {
  if (!missionHasDefinedPickupTime(mission.scheduled_at)) {
    return null;
  }
  const delay = Number(mission.assignment_pickup_delay_minutes);
  if (Number.isFinite(delay) && delay > 0) {
    return `+${Math.round(delay)} min`;
  }
  if (mission.status === "en_route" || mission.status === "in_progress") {
    return "En route";
  }

  // Fallback ETA from API hints (web-aligned payloads may expose pickup/current ETA).
  const etaCandidate = (
    mission as CompanyDispatchMission & {
      pickup_eta?: string | null;
      current_eta?: string | null;
    }
  ).pickup_eta || (
    mission as CompanyDispatchMission & {
      pickup_eta?: string | null;
      current_eta?: string | null;
    }
  ).current_eta;

  if (typeof etaCandidate === "string" && etaCandidate.trim()) {
    const etaTs = Date.parse(etaCandidate.trim());
    if (Number.isFinite(etaTs)) {
      const diffMin = Math.round((etaTs - Date.now()) / 60_000);
      if (diffMin <= 3) return "Imminent";
      if (diffMin > 3) return `~${diffMin} min`;
    }
  }

  // Last-resort fallback: use scheduled time when mission is upcoming/assigned.
  if (mission.status === "assigned" || mission.status === "accepted" || mission.status === "pending") {
    const scheduledTs = toEpoch(mission.scheduled_at);
    if (scheduledTs > 0) {
      const diffMin = Math.round((scheduledTs - Date.now()) / 60_000);
      if (diffMin <= 3) return "Imminent";
      if (diffMin > 3) return `~${diffMin} min`;
    }
  }

  return null;
}

export function isMissionDelayed(mission: CompanyDispatchMission, nowMs = Date.now()): boolean {
  return resolveMissionUiStatus(mission, nowMs).tone === "delayed";
}

export const IN_FLIGHT_MISSION_STATUSES: CompanyDispatchMissionStatus[] = ["en_route", "in_progress"];
