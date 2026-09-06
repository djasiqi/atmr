import type { QueryClient } from "@tanstack/react-query";
import { contextScopedKey } from "../../../core/cache/contextCache";
import {
  getCompanyRideDetail,
  normalizeDispatchMission,
} from "../api/companyApi";
import type {
  CompanyDispatchMission,
  CompanyDispatchMissionListResponse,
} from "../api/contracts";
import { companyQueryKeys } from "../companyQueryKeys";
import { areDispatchMissionsContentEqual } from "./dispatchMissionListReconcile";
import { dispatchMissionsQueryKey } from "./prefetchAdjacentDispatchMissions";
import {
  markAuthoritativeDispatchSync,
  setStickyRidesFetchReason,
  wasRecentlyAuthoritativelySynced,
  type RidesFetchReason,
} from "./ridesFetchReason";

const RECONCILE_DEDUP_MS = 5_000;
const recentMissionReconcile = new Map<string, number>();

export function resetDispatchMissionCachePatchForTests(): void {
  recentMissionReconcile.clear();
}

export function rideDetailsQueryKey(contextId: string, missionId: number): unknown[] {
  return contextScopedKey(contextId, [
    ...companyQueryKeys.rideDetails(contextId, missionId),
  ] as unknown[]);
}

export function isDispatchMissionsQueryKey(key: unknown, contextId?: string): boolean {
  if (!Array.isArray(key)) return false;
  if (contextId != null && (key[0] !== "ctx" || key[1] !== contextId)) return false;
  if (!key.includes("dispatch") || !key.includes("missions")) return false;
  if (key.includes("ride-details")) return false;
  return true;
}

export function dateFromMissionsQueryKey(key: unknown): string | null {
  if (!Array.isArray(key)) return null;
  const last = key[key.length - 1];
  return typeof last === "string" && /^\d{4}-\d{2}-\d{2}$/.test(last) ? last : null;
}

function missionReconcileKey(contextId: string, missionId: number): string {
  return `${contextId}:${missionId}`;
}

function markMissionReconciled(contextId: string, missionId: number): void {
  recentMissionReconcile.set(missionReconcileKey(contextId, missionId), Date.now());
  if (recentMissionReconcile.size > 64) {
    const now = Date.now();
    for (const [entry, stamp] of recentMissionReconcile) {
      if (now - stamp > RECONCILE_DEDUP_MS * 2) recentMissionReconcile.delete(entry);
    }
  }
  markAuthoritativeDispatchSync();
}

export function wasMissionRecentlyReconciled(contextId: string, missionId: number): boolean {
  const previous = recentMissionReconcile.get(missionReconcileKey(contextId, missionId));
  return previous != null && Date.now() - previous < RECONCILE_DEDUP_MS;
}

export function extractRideDetailRecord(
  payload: unknown,
  missionId: number
): Record<string, unknown> | null {
  if (!payload || typeof payload !== "object") return null;
  const root = payload as Record<string, unknown>;
  const directCandidate = root.summary ?? root.data ?? root.item ?? root.ride ?? root.mission;
  if (directCandidate && typeof directCandidate === "object") {
    const row = directCandidate as Record<string, unknown>;
    const id = Number(row.mission_id ?? row.booking_id ?? row.id);
    if (Number.isFinite(id) && id === missionId) return row;
  }
  const rows =
    (Array.isArray(root.items) && root.items) ||
    (Array.isArray(root.missions) && root.missions) ||
    (Array.isArray(root.data) && root.data) ||
    [];
  for (const entry of rows) {
    if (!entry || typeof entry !== "object") continue;
    const row = entry as Record<string, unknown>;
    const id = Number(row.mission_id ?? row.booking_id ?? row.id);
    if (Number.isFinite(id) && id === missionId) return row;
  }
  const selfId = Number(root.mission_id ?? root.booking_id ?? root.id);
  if (Number.isFinite(selfId) && selfId === missionId) return root;
  return null;
}

export function listCachedDispatchDays(
  queryClient: QueryClient,
  contextId: string
): {
  queryKey: unknown[];
  date: string;
  data: CompanyDispatchMissionListResponse;
  observerCount: number;
}[] {
  const out: {
    queryKey: unknown[];
    date: string;
    data: CompanyDispatchMissionListResponse;
    observerCount: number;
  }[] = [];
  for (const query of queryClient.getQueryCache().findAll({
    predicate: (candidate) => isDispatchMissionsQueryKey(candidate.queryKey, contextId),
  })) {
    const date = dateFromMissionsQueryKey(query.queryKey);
    const data = query.state.data as CompanyDispatchMissionListResponse | undefined;
    if (!date || !data?.missions) continue;
    out.push({
      queryKey: query.queryKey as unknown[],
      date,
      data,
      observerCount: query.getObserversCount(),
    });
  }
  return out;
}

/**
 * Remplace #mission dans les journées qui la contiennent.
 * Conserve total / loaded / is_complete / pagination_error / next_page.
 * Les missions non touchées gardent leur référence.
 */
export function patchMissionInCachedDays(
  queryClient: QueryClient,
  contextId: string,
  incoming: CompanyDispatchMission
): { patchedDays: string[]; patched: boolean } {
  const patchedDays: string[] = [];
  for (const day of listCachedDispatchDays(queryClient, contextId)) {
    const index = day.data.missions.findIndex((row) => row.mission_id === incoming.mission_id);
    if (index < 0) continue;
    const previous = day.data.missions[index];
    const nextMission = areDispatchMissionsContentEqual(previous, incoming) ? previous : incoming;
    if (nextMission === previous) {
      patchedDays.push(day.date);
      continue;
    }
    const nextMissions = day.data.missions.map((row, rowIndex) =>
      rowIndex === index ? nextMission : row
    );
    queryClient.setQueryData<CompanyDispatchMissionListResponse>(day.queryKey, {
      ...day.data,
      missions: nextMissions,
      refreshed_at: new Date().toISOString(),
    });
    patchedDays.push(day.date);
  }
  return { patchedDays, patched: patchedDays.length > 0 };
}

export function patchRideDetailsIfPresent(
  queryClient: QueryClient,
  contextId: string,
  missionId: number,
  detail: Record<string, unknown>
): boolean {
  const queryKey = rideDetailsQueryKey(contextId, missionId);
  const existing = queryClient.getQueryData(queryKey);
  if (existing == null) return false;
  queryClient.setQueryData(queryKey, detail);
  return true;
}

export async function refetchExactDispatchDay(
  queryClient: QueryClient,
  contextId: string,
  date: string,
  reason: RidesFetchReason
): Promise<void> {
  setStickyRidesFetchReason(reason);
  await queryClient.refetchQueries({
    queryKey: dispatchMissionsQueryKey(contextId, date),
    exact: true,
  });
  markAuthoritativeDispatchSync();
}

/** J observé uniquement — J±1 prefetch (0 observer) n’est pas relancé. */
export async function refetchObservedDispatchDays(
  queryClient: QueryClient,
  contextId: string,
  reason: RidesFetchReason
): Promise<number> {
  const observed = listCachedDispatchDays(queryClient, contextId).filter(
    (day) => day.observerCount > 0
  );
  if (observed.length === 0) return 0;
  setStickyRidesFetchReason(reason);
  await queryClient.refetchQueries({
    type: "active",
    predicate: (query) =>
      isDispatchMissionsQueryKey(query.queryKey, contextId) &&
      dateFromMissionsQueryKey(query.queryKey) != null,
  });
  markAuthoritativeDispatchSync();
  return observed.length;
}

function scheduledDateOf(mission: CompanyDispatchMission): string | null {
  const raw = mission.scheduled_at;
  if (typeof raw !== "string" || raw.length < 10) return null;
  const iso = raw.slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(iso) ? iso : null;
}

/**
 * Récupère la mission autoritaire puis patche J + ride-details.
 * Fallback : refetch EXACTEMENT la journée connue, jamais la famille rides.
 */
export async function reconcileAuthoritativeMission(
  queryClient: QueryClient,
  contextId: string,
  missionId: number,
  reason: RidesFetchReason
): Promise<"patched" | "day_refetch" | "observed_refetch" | "skipped"> {
  if (wasMissionRecentlyReconciled(contextId, missionId)) {
    return "skipped";
  }
  try {
    const payload = await getCompanyRideDetail({ contextId, missionId });
    const detail = extractRideDetailRecord(payload, missionId);
    const mission = detail ? normalizeDispatchMission(detail) : normalizeDispatchMission(payload);
    if (mission) {
      const { patched } = patchMissionInCachedDays(queryClient, contextId, mission);
      if (detail) {
        patchRideDetailsIfPresent(queryClient, contextId, missionId, detail);
      }
      markMissionReconciled(contextId, missionId);
      if (patched) return "patched";
      const date = scheduledDateOf(mission);
      if (date) {
        await refetchExactDispatchDay(queryClient, contextId, date, reason);
        return "day_refetch";
      }
    }
  } catch {
    // Fallback ciblé ci-dessous.
  }
  await refetchObservedDispatchDays(queryClient, contextId, reason);
  return "observed_refetch";
}

export function shouldSkipFocusRefetchForDispatchDay(): boolean {
  return wasRecentlyAuthoritativelySynced();
}
