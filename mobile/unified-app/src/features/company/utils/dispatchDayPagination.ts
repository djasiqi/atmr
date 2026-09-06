import type {
  CompanyDispatchMission,
  CompanyDispatchMissionListResponse,
} from "../api/contracts";
import { reconcileDispatchMissionList } from "./dispatchMissionListReconcile";

/** Taille de page de la journée ouverte — jamais 500 (plafond serveur = 100). */
export const DISPATCH_DAY_PAGE_SIZE = 50;

export type DispatchSearchPresentation =
  | { kind: "idle" }
  | { kind: "hits" }
  | { kind: "pending"; remaining: number }
  | { kind: "none" };

function finitePositive(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function finiteNonNegative(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : fallback;
}

export function remainingUnloadedCount(
  day: Pick<CompanyDispatchMissionListResponse, "loaded" | "total">
): number {
  return Math.max(0, day.total - day.loaded);
}

export function shouldFetchNextDayPage(
  day: CompanyDispatchMissionListResponse | undefined,
  completeDay: boolean
): boolean {
  if (!completeDay || !day) return false;
  const normalized = ensureDayPagination(day);
  if (normalized.pagination_error) return false;
  if (normalized.is_complete) return false;
  return normalized.loaded < normalized.total;
}

/** Complète les champs pagination si un cache ancien n’en a pas. */
export function ensureDayPagination(
  data: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  const missions = Array.isArray(data.missions) ? data.missions : [];
  const pageSize = finitePositive(data.page_size, DISPATCH_DAY_PAGE_SIZE);
  const total = finiteNonNegative(data.total, missions.length);
  const loaded = finiteNonNegative(data.loaded, missions.length);
  const page = finitePositive(data.page, 1);
  const nextPage = finitePositive(data.next_page, page + 1);
  const isComplete = data.is_complete ?? loaded >= total;
  return {
    ...data,
    missions,
    total,
    page_size: pageSize,
    loaded,
    page,
    next_page: nextPage,
    is_complete: isComplete,
    pagination_error: data.pagination_error === true,
  };
}

function dedupeMissionsById(missions: CompanyDispatchMission[]): CompanyDispatchMission[] {
  const seen = new Set<number>();
  const out: CompanyDispatchMission[] = [];
  for (const mission of missions) {
    if (seen.has(mission.mission_id)) continue;
    seen.add(mission.mission_id);
    out.push(mission);
  }
  return out;
}

/**
 * Fusionne une page serveur dans la journée déjà chargée.
 * Ordre = pages 1..N-1 + page N + queue éventuelle ; dédup par `mission_id`.
 * Une page d’une autre date / autre contexte remplace (jamais de contamination).
 */
export function applyDayPage(
  previous: CompanyDispatchMissionListResponse | undefined,
  incoming: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  const page = finitePositive(incoming.page, 1);
  const pageSize = finitePositive(incoming.page_size, DISPATCH_DAY_PAGE_SIZE);
  const total = finiteNonNegative(incoming.total, incoming.missions.length);
  const incomingMissions = incoming.missions;

  if (
    previous &&
    ((previous.date && incoming.date && previous.date !== incoming.date) ||
      previous.context_id !== incoming.context_id)
  ) {
    return previous;
  }

  const prior = previous?.missions ?? [];
  const start = (page - 1) * pageSize;
  const expectedOnThisPage = Math.min(pageSize, Math.max(0, total - start));
  if (incomingMissions.length === 0 && expectedOnThisPage > 0) {
    const kept = previous ?? finalizeDayPage(incoming, [], page, pageSize, total);
    return {
      ...kept,
      total,
      page_size: pageSize,
      loaded: kept.missions.length,
      page,
      next_page: page,
      is_complete: false,
      pagination_error: true,
    };
  }
  const incomingIds = new Set(incomingMissions.map((mission) => mission.mission_id));
  const head = prior.slice(0, start).filter((mission) => !incomingIds.has(mission.mission_id));
  const pageReconciled = reconcileDispatchMissionList(prior, incomingMissions);
  const loadedAfterPage = start + pageReconciled.length;
  const moreExpected = loadedAfterPage < total;
  const tail = moreExpected
    ? prior.slice(start + pageReconciled.length).filter((mission) => !incomingIds.has(mission.mission_id))
    : [];
  const merged = dedupeMissionsById([...head, ...pageReconciled, ...tail]);
  return finalizeDayPage(incoming, merged, page, pageSize, total);
}

function finalizeDayPage(
  incoming: CompanyDispatchMissionListResponse,
  missions: CompanyDispatchMission[],
  page: number,
  pageSize: number,
  total: number
): CompanyDispatchMissionListResponse {
  const loaded = missions.length;
  const emptyPageWhileIncomplete = incoming.missions.length === 0 && loaded < total;
  const isComplete = loaded >= total && !emptyPageWhileIncomplete;
  return {
    ...incoming,
    missions,
    total,
    page_size: pageSize,
    loaded,
    page,
    next_page: emptyPageWhileIncomplete ? page : page + 1,
    is_complete: isComplete,
    pagination_error: emptyPageWhileIncomplete,
  };
}

export function markDayPaginationError(
  day: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  return {
    ...day,
    is_complete: false,
    pagination_error: true,
  };
}

export function clearDayPaginationError(
  day: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  return {
    ...day,
    pagination_error: false,
    is_complete: day.loaded >= day.total,
  };
}

/**
 * Recherche locale : hits immédiats sur le chargé.
 * « Aucun résultat » seulement si la journée est complète.
 */
export function resolveDispatchSearchPresentation(input: {
  query: string;
  hitCount: number;
  loaded: number;
  total: number;
  isComplete: boolean;
}): DispatchSearchPresentation {
  const needle = input.query.trim();
  if (!needle) return { kind: "idle" };
  if (input.hitCount > 0) return { kind: "hits" };
  if (!input.isComplete) {
    return { kind: "pending", remaining: Math.max(0, input.total - input.loaded) };
  }
  return { kind: "none" };
}
