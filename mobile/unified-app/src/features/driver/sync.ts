import { QueryClient } from "@tanstack/react-query";
import { getDriverMissionsSince } from "./api";
import { getDriverMissions } from "./api/driverHttp";
import { driverOfflineQueue } from "./offlineQueue";
import { driverQueryKeys } from "./queryKeys";
import { DriverMission } from "./types";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { missionRuntimeManager } from "./services/missionRuntimeManager";
import { setActiveMissionFromList } from "./services/missionState";
import { realtimeManager } from "../../core/realtime/realtimeManager";
import { isFeatureEnabled } from "../../core/featureFlags/registry";

const LAST_SYNC_KEY = "driver_last_sync_at_v1";
const MAX_INCREMENTAL_WINDOW_MS = Number(
  process.env.EXPO_PUBLIC_REALTIME_MAX_INCREMENTAL_WINDOW_MS ?? "900000"
);

async function readLastSync(): Promise<string | null> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    return await storage.default.getItem(LAST_SYNC_KEY);
  } catch {
    return null;
  }
}

async function writeLastSync(iso: string): Promise<void> {
  try {
    const storage = await import("@react-native-async-storage/async-storage");
    await storage.default.setItem(LAST_SYNC_KEY, iso);
  } catch {
    // Best effort.
  }
}

export async function reconcileDriverMissions(
  queryClient: QueryClient,
  contextId: string
): Promise<{ missions: DriverMission[]; queue: { sent: number; dropped: number; failed: number } }> {
  const lastSyncIso = await readLastSync();
  const since = lastSyncIso ?? new Date(Date.now() - 5 * 60_000).toISOString();
  const sinceMs = Date.parse(since);
  const sinceAgeMs = Number.isFinite(sinceMs) ? Date.now() - sinceMs : Number.POSITIVE_INFINITY;
  const useFullRefetchGuard =
    isFeatureEnabled("realtime_adaptive_polling_enabled") && sinceAgeMs > MAX_INCREMENTAL_WINDOW_MS;
  const missions = useFullRefetchGuard
    ? await getDriverMissions()
    : await getDriverMissionsSince(since);
  realtimeManager.setTransportAuthority("reconcile", useFullRefetchGuard ? "full_refetch_guard" : "since");
  emitDriverTelemetry(
    useFullRefetchGuard ? "realtime.reconcile.full_refetch_guarded" : "realtime.reconcile.since",
    {
      source: "driver.sync.reconcile",
      context_id: contextId,
      mission_reconcile_since_total: useFullRefetchGuard ? 0 : 1,
      since_age_ms: Number.isFinite(sinceAgeMs) ? sinceAgeMs : null,
      max_incremental_window_ms: MAX_INCREMENTAL_WINDOW_MS,
    }
  );
  await queryClient.setQueryData(driverQueryKeys.missions(contextId), (previous: unknown) => {
    const prev = Array.isArray(previous) ? (previous as DriverMission[]) : [];
    const previousById = new Map<number, DriverMission>();
    prev.forEach((mission) => {
      previousById.set(mission.id, mission);
    });
    let driftCount = 0;
    missions.forEach((mission) => {
      const local = previousById.get(mission.id);
      if (!local) return;
      const localStatus = String(local.status ?? "");
      const remoteStatus = String(mission.status ?? "");
      if (localStatus !== remoteStatus) {
        driftCount += 1;
      }
    });
    if (driftCount > 0) {
      emitDriverTelemetry("realtime.drift.detected", {
        source: "driver.sync.reconcile",
        context_id: contextId,
        drift_count: driftCount,
      });
    }
    if (missions.length === 0) return prev;
    const byId = new Map<number, DriverMission>();
    prev.forEach((mission) => {
      byId.set(mission.id, mission);
    });
    missions.forEach((mission) => {
      byId.set(mission.id, mission);
      missionRuntimeManager.registerSnapshot(mission.id, String(mission.updated_at ?? null));
    });
    return Array.from(byId.values());
  });
  if (missions.length > 0) {
    const updatedAtMs = Date.parse(String(missions[0]?.updated_at ?? ""));
    if (Number.isFinite(updatedAtMs)) {
      emitDriverTelemetry("realtime.mission.freshness", {
        source: "driver.sync.reconcile",
        context_id: contextId,
        realtime_transport_mode: "reconcile",
        mission_state_freshness_ms: Math.max(0, Date.now() - updatedAtMs),
      });
    }
  }
  await writeLastSync(new Date().toISOString());
  await setActiveMissionFromList(missions);
  const queue = await driverOfflineQueue.flush();
  return { missions, queue };
}

