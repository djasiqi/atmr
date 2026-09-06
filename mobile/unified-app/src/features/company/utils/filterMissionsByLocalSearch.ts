import type { CompanyDispatchMission } from "../api/contracts";
import { missionBelongsToSelectedDay } from "./companyDateUtils";

/** Neutralise casse et accents pour une recherche locale instantanée. */
export function normalizeMissionSearchText(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim();
}

const normalizedSearchIndexByMission = new WeakMap<CompanyDispatchMission, string>();

function collectMissionSearchHaystack(mission: CompanyDispatchMission): string {
  const identity = mission.identity;
  const parts: (string | null | undefined)[] = [
    mission.client_name,
    identity?.passenger?.name,
    mission.pickup_label,
    mission.dropoff_label,
    mission.driver_name,
    identity?.source?.name,
    identity?.source?.code,
    identity?.ownership?.owner_company_name,
    identity?.execution?.executing_company_name,
    identity?.upstream?.name,
    identity?.requester?.name,
    mission.partner_company_name,
  ];
  if (Array.isArray(mission.search_index)) {
    parts.push(...mission.search_index);
  }
  return parts
    .filter((part): part is string => typeof part === "string" && part.trim().length > 0)
    .join(" ");
}

/** Index normalisé stable par objet mission — recalculé seulement si la référence change. */
export function getMissionNormalizedSearchIndex(mission: CompanyDispatchMission): string {
  const cached = normalizedSearchIndexByMission.get(mission);
  if (cached !== undefined) return cached;
  const index = normalizeMissionSearchText(collectMissionSearchHaystack(mission));
  normalizedSearchIndexByMission.set(mission, index);
  return index;
}

/** Filtre local de la journée déjà chargée — aucun appel réseau. */
export function filterMissionsByLocalSearch(
  missions: CompanyDispatchMission[],
  query: string
): CompanyDispatchMission[] {
  const needle = normalizeMissionSearchText(query);
  if (!needle) return missions;
  return missions.filter((mission) => getMissionNormalizedSearchIndex(mission).includes(needle));
}

/**
 * Recherche limitée à la date affichée : jamais de course d’un autre jour.
 */
export function filterDayMissionsForLocalSearch(
  missions: CompanyDispatchMission[],
  selectedDate: string,
  query: string
): CompanyDispatchMission[] {
  const ofDay = missions.filter((mission) => missionBelongsToSelectedDay(mission, selectedDate));
  return filterMissionsByLocalSearch(ofDay, query);
}
