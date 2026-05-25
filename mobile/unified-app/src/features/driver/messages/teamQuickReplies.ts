import { filterNextMissionsOnly } from "../domain/missionGrouping";
import {
  getClientFormalAddress,
  getMissionClientDisplayName,
} from "../domain/missionDisplay";
import { getDriverStatusUx, resolveDriverStatusForUx } from "../statusDictionary";
import type { DriverMission, DriverMissionStatus } from "../types";

export type TeamQuickReply = {
  id: string;
  label: string;
  content: string;
};

const FALLBACK_TEAM_QUICK_REPLIES: TeamQuickReply[] = [
  { id: "available", label: "✅ Disponible", content: "Je suis disponible" },
  { id: "onsite", label: "📍 Sur place", content: "Je suis sur place" },
  { id: "delay5", label: "🕒 Retard 5 min", content: "Retard 5 min" },
];

export function selectActiveDriverMission(
  missions: DriverMission[] | undefined
): DriverMission | null {
  if (!Array.isArray(missions) || missions.length === 0) return null;
  const nextScope = filterNextMissionsOnly(missions);
  if (nextScope.length > 0) return nextScope[0] ?? null;
  const firstNonTerminal = missions.find((mission) => {
    const ux = getDriverStatusUx(typeof mission.status === "string" ? mission.status : null);
    return !ux.terminal;
  });
  return firstNonTerminal ?? null;
}

function repliesForStatus(
  status: DriverMissionStatus,
  formal: string
): TeamQuickReply[] {
  switch (status) {
    case "IN_PROGRESS":
      return [
        {
          id: "onboard",
          label: `👤 ${formal} à bord`,
          content: `${formal} à bord`,
        },
        {
          id: "completed",
          label: `✅ ${formal}`,
          content: `${formal} — course terminée`,
        },
        {
          id: "delay5",
          label: "🕒 Retard 5 min",
          content: `Retard 5 min — ${formal}`,
        },
        {
          id: "delay10",
          label: "🕒 Retard 10 min",
          content: `Retard 10 min — ${formal}`,
        },
      ];
    case "EN_ROUTE":
      return [
        {
          id: "arrive",
          label: "🚗 J'arrive",
          content: `J'arrive — ${formal}`,
        },
        {
          id: "delay5",
          label: "🕒 Retard 5 min",
          content: `Retard 5 min — ${formal}`,
        },
        {
          id: "delay10",
          label: "🕒 Retard 10 min",
          content: `Retard 10 min — ${formal}`,
        },
        {
          id: "onsite",
          label: "📍 Sur place",
          content: `Sur place — ${formal}`,
        },
      ];
    case "ARRIVED":
      return [
        {
          id: "arrive",
          label: "🚗 J'arrive",
          content: `J'arrive — ${formal}`,
        },
        {
          id: "onboard",
          label: `👤 ${formal} à bord`,
          content: `${formal} à bord`,
        },
        {
          id: "waiting",
          label: "⏳ En attente",
          content: `En attente — ${formal}`,
        },
        {
          id: "delay5",
          label: "🕒 Retard 5 min",
          content: `Retard 5 min — ${formal}`,
        },
      ];
    case "ASSIGNED":
      return [
        {
          id: "enroute",
          label: "🚗 En route",
          content: `En route — ${formal}`,
        },
        {
          id: "arrive",
          label: "🚗 J'arrive",
          content: `J'arrive — ${formal}`,
        },
        {
          id: "delay5",
          label: "🕒 Retard 5 min",
          content: `Retard 5 min — ${formal}`,
        },
        {
          id: "onsite",
          label: "📍 Sur place",
          content: `Sur place — ${formal}`,
        },
      ];
    default:
      return FALLBACK_TEAM_QUICK_REPLIES;
  }
}

/** Suggestions canal équipe selon la mission active du chauffeur. */
export function buildTeamQuickReplies(activeMission: DriverMission | null): TeamQuickReply[] {
  if (!activeMission) return FALLBACK_TEAM_QUICK_REPLIES;
  const status = resolveDriverStatusForUx(activeMission.status);
  const ux = getDriverStatusUx(activeMission.status);
  if (ux.terminal) return FALLBACK_TEAM_QUICK_REPLIES;
  const formal = getClientFormalAddress(activeMission);
  if (formal.includes("Mission #")) {
    const name = getMissionClientDisplayName(activeMission);
    return repliesForStatus(status, name);
  }
  return repliesForStatus(status, formal);
}
