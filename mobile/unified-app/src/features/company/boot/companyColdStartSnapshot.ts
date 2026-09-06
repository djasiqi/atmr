import type { QueryClient } from "@tanstack/react-query";
import { getItem, setItem } from "../../../core/storage/typedStorage";
import { STORAGE_KEYS } from "../../../core/storage/storageKeys";
import { contextScopedKey } from "../../../core/cache/contextCache";
import type {
  CompanyDispatchMissionListResponse,
  CompanyDispatchRealtimeDashboard,
  CompanyDriverLiveLocation,
  CompanyDriverLiveLocationResponse,
} from "../api/contracts";
import type { CompanyDispatchStatusResponse } from "../api/companyApi";
import { companyQueryKeys } from "../companyQueryKeys";
import { DISPATCH_DAY_PAGE_SIZE } from "../utils/dispatchDayPagination";

export const COMPANY_COLD_START_SNAPSHOT_VERSION = 1 as const;
export const COMPANY_COLD_START_SNAPSHOT_KEY = STORAGE_KEYS.COMPANY_COLD_START_SNAPSHOT;
export const COMPANY_COLD_START_MAX_MISSIONS = DISPATCH_DAY_PAGE_SIZE;
export const COMPANY_COLD_START_MAX_DRIVERS = 120;

export type CompanyColdStartSliceMeta = {
  cached_at: string;
  server_refreshed_at: string;
};

export type CompanyColdStartSnapshot = {
  version: typeof COMPANY_COLD_START_SNAPSHOT_VERSION;
  context_id: string;
  date: string;
  cached_at: string;
  missions: CompanyDispatchMissionListResponse | null;
  missions_meta: CompanyColdStartSliceMeta | null;
  dashboard: CompanyDispatchRealtimeDashboard | null;
  dashboard_meta: CompanyColdStartSliceMeta | null;
  roster: CompanyDriverLiveLocationResponse | null;
  roster_meta: CompanyColdStartSliceMeta | null;
  dispatch_status: CompanyDispatchStatusResponse | null;
  dispatch_status_meta: CompanyColdStartSliceMeta | null;
};

export type CompanyColdStartPersistInput = {
  contextId: string;
  date: string;
  nowIso?: string;
  missions?: CompanyDispatchMissionListResponse | null;
  dashboard?: CompanyDispatchRealtimeDashboard | null;
  roster?: CompanyDriverLiveLocationResponse | null;
  dispatchStatus?: CompanyDispatchStatusResponse | null;
};

const memoryByContext = new Map<string, CompanyColdStartSnapshot>();

function toIso(value: string | undefined, fallbackIso: string): string {
  if (value && Number.isFinite(Date.parse(value))) return value;
  return fallbackIso;
}

function sliceMeta(serverRefreshedAt: string | undefined, nowIso: string): CompanyColdStartSliceMeta {
  return {
    cached_at: nowIso,
    server_refreshed_at: toIso(serverRefreshedAt, nowIso),
  };
}

function serverRefreshedAtMs(meta: CompanyColdStartSliceMeta | null): number {
  if (!meta) return 0;
  const parsed = Date.parse(meta.server_refreshed_at);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Âge GPS = timestamp réel. `last_seen_seconds` est relatif au fetch :
 * le garder ferait rajeunir une position au cold start.
 */
export function persistDriverLocationForDisk(
  driver: CompanyDriverLiveLocation
): CompanyDriverLiveLocation {
  const { last_seen_seconds: _relative, ...absolute } = driver;
  void _relative;
  return absolute;
}

export function boundMissionsForDisk(
  data: CompanyDispatchMissionListResponse
): CompanyDispatchMissionListResponse {
  const missions = data.missions.slice(0, COMPANY_COLD_START_MAX_MISSIONS);
  const loaded = missions.length;
  const wasTrimmed = data.missions.length > COMPANY_COLD_START_MAX_MISSIONS;
  return {
    ...data,
    missions,
    loaded,
    is_complete: wasTrimmed ? false : data.is_complete,
    next_page: wasTrimmed ? 2 : data.next_page,
    page_size: data.page_size || DISPATCH_DAY_PAGE_SIZE,
  };
}

export function boundRosterForDisk(
  data: CompanyDriverLiveLocationResponse
): CompanyDriverLiveLocationResponse {
  return {
    ...data,
    locations: data.locations.slice(0, COMPANY_COLD_START_MAX_DRIVERS).map(persistDriverLocationForDisk),
  };
}

export function buildCompanyColdStartSnapshot(
  input: CompanyColdStartPersistInput
): CompanyColdStartSnapshot {
  const nowIso = input.nowIso ?? new Date().toISOString();
  const missions = input.missions ? boundMissionsForDisk(input.missions) : null;
  const roster = input.roster ? boundRosterForDisk(input.roster) : null;
  return {
    version: COMPANY_COLD_START_SNAPSHOT_VERSION,
    context_id: input.contextId,
    date: input.date,
    cached_at: nowIso,
    missions,
    missions_meta: missions ? sliceMeta(missions.refreshed_at, nowIso) : null,
    dashboard: input.dashboard ?? null,
    dashboard_meta: input.dashboard ? sliceMeta(input.dashboard.refreshed_at, nowIso) : null,
    roster,
    roster_meta: roster ? sliceMeta(roster.refreshed_at, nowIso) : null,
    dispatch_status: input.dispatchStatus ?? null,
    dispatch_status_meta: input.dispatchStatus
      ? sliceMeta(input.dispatchStatus.refreshed_at, nowIso)
      : null,
  };
}

export function isCompanyColdStartSnapshot(
  value: unknown
): value is CompanyColdStartSnapshot {
  if (!value || typeof value !== "object") return false;
  const row = value as CompanyColdStartSnapshot;
  return (
    row.version === COMPANY_COLD_START_SNAPSHOT_VERSION &&
    typeof row.context_id === "string" &&
    row.context_id.length > 0 &&
    typeof row.date === "string" &&
    typeof row.cached_at === "string"
  );
}

function markQueryStale(queryClient: QueryClient, queryKey: unknown[], dataUpdatedAt: number): void {
  const query = queryClient.getQueryCache().find({ queryKey, exact: true });
  query?.setState({ dataUpdatedAt: Math.min(dataUpdatedAt, Date.now() - 1) });
}

export function applyCompanyColdStartSnapshot(
  queryClient: QueryClient,
  snapshot: CompanyColdStartSnapshot
): void {
  const { context_id: contextId, date } = snapshot;
  if (snapshot.missions) {
    const key = contextScopedKey(contextId, [...companyQueryKeys.missions(contextId, date)] as unknown[]);
    queryClient.setQueryData(key, snapshot.missions);
    markQueryStale(queryClient, key, serverRefreshedAtMs(snapshot.missions_meta));
  }
  if (snapshot.dashboard) {
    const key = contextScopedKey(contextId, [...companyQueryKeys.dashboard(contextId), date] as unknown[]);
    queryClient.setQueryData(key, snapshot.dashboard);
    markQueryStale(queryClient, key, serverRefreshedAtMs(snapshot.dashboard_meta));
  }
  if (snapshot.roster) {
    const key = contextScopedKey(contextId, [...companyQueryKeys.driversLocations(contextId)] as unknown[]);
    queryClient.setQueryData(key, snapshot.roster);
    markQueryStale(queryClient, key, serverRefreshedAtMs(snapshot.roster_meta));
  }
  if (snapshot.dispatch_status) {
    const key = contextScopedKey(contextId, [
      ...companyQueryKeys.root,
      "dispatch-status",
      contextId,
      date,
    ] as unknown[]);
    queryClient.setQueryData(key, snapshot.dispatch_status);
    markQueryStale(queryClient, key, serverRefreshedAtMs(snapshot.dispatch_status_meta));
  }
}

export function peekCompanyColdStartSnapshot(contextId: string): CompanyColdStartSnapshot | null {
  return memoryByContext.get(contextId) ?? null;
}

export async function preloadCompanyColdStartSnapshot(
  contextId: string
): Promise<CompanyColdStartSnapshot | null> {
  const memory = memoryByContext.get(contextId);
  if (memory) return memory;
  const stored = await getItem<CompanyColdStartSnapshot>(COMPANY_COLD_START_SNAPSHOT_KEY);
  if (!isCompanyColdStartSnapshot(stored) || stored.context_id !== contextId) return null;
  memoryByContext.set(contextId, stored);
  return stored;
}

export async function hydrateCompanyColdStartSnapshot(
  queryClient: QueryClient,
  contextId: string | null
): Promise<boolean> {
  if (!contextId) return false;
  const snapshot = await preloadCompanyColdStartSnapshot(contextId);
  if (!snapshot) return false;
  applyCompanyColdStartSnapshot(queryClient, snapshot);
  return true;
}

export async function persistCompanyColdStartSnapshot(
  input: CompanyColdStartPersistInput
): Promise<CompanyColdStartSnapshot> {
  const snapshot = buildCompanyColdStartSnapshot(input);
  memoryByContext.set(input.contextId, snapshot);
  await setItem(COMPANY_COLD_START_SNAPSHOT_KEY, snapshot);
  return snapshot;
}

export function resetCompanyColdStartSnapshotForTests(): void {
  memoryByContext.clear();
}
