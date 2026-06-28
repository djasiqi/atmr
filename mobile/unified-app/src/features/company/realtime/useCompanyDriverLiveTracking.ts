import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  configureGpsMetricsSampling,
  recordGpsBatchCoalesced,
  recordGpsBatchFlush,
  recordMapSetState,
  shouldSampleGpsMetricEvent,
} from "../../../core/observability/realtimeMetrics";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { useActiveCompanyContextId, useCompanyDriversLocationsSnapshotQuery } from "../hooks";
import { mergeDriverLocationPreserveReference } from "./driverLiveLocationMerge";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  resolveFlushDelayMs,
  resolveGpsEventFlushPriority,
  type GpsFlushPriority,
  getEffectiveMaxBatchAgeMs,
} from "./gpsFlushScheduler";
import { REALTIME_FLUSH_MS } from "./gpsFlushConstants";
import { resolveRealtimeFlushMs } from "./companyMapRuntimeConfig";

export { REALTIME_FLUSH_MS };
export const MAX_BATCH_AGE_MS = 1_000;
export { resolveRealtimeFlushMs as REALTIME_FLUSH_MS_RESOLVED };
export const MAP_SILENCE_RESYNC_MS = 120_000;
const STALE_SECONDS_THRESHOLD = 120;

function toEpoch(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

type DriverRealtimePayload = Partial<CompanyDriverLiveLocation> & {
  driver_id?: number | string;
  driver_name?: string | null;
  full_name?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  lat?: number;
  lon?: number;
  lng?: number;
  last_seen_seconds?: number | null;
  location_status?: "live" | "recent" | "stale" | "offline" | "last_known" | null;
  mission_status?: string | null;
  status?: string | null;
  presence_status?: string | null;
  tracking_display_status?: string | null;
  device_health?: CompanyDriverLiveLocation["device_health"];
  recorded_at?: string | null;
  received_at?: string | null;
};

function extractLatitude(payload: DriverRealtimePayload): number {
  return Number(payload.latitude ?? payload.lat);
}

function extractLongitude(payload: DriverRealtimePayload): number {
  // Canonique progressif: `lng`; compat transitoire: `lon` / `longitude`
  return Number(payload.lng ?? payload.lon ?? payload.longitude);
}

function normalizeLocationStatus(
  status: DriverRealtimePayload["location_status"],
  lastSeenSeconds: number | null
): CompanyDriverLiveLocation["location_status"] {
  if (status) return status;
  if (typeof lastSeenSeconds === "number" && Number.isFinite(lastSeenSeconds)) {
    return lastSeenSeconds > STALE_SECONDS_THRESHOLD ? "stale" : "live";
  }
  return null;
}

export function normalizeRealtimeLocation(payload: DriverRealtimePayload): CompanyDriverLiveLocation | null {
  const driverId = Number(payload.driver_id);
  if (!Number.isFinite(driverId)) return null;
  const latitude = extractLatitude(payload);
  const longitude = extractLongitude(payload);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  const lastSeenSeconds =
    typeof payload.last_seen_seconds === "number" && Number.isFinite(payload.last_seen_seconds)
      ? payload.last_seen_seconds
      : null;
  const firstName =
    typeof payload.first_name === "string" && payload.first_name.trim()
      ? payload.first_name.trim()
      : null;
  const lastName =
    typeof payload.last_name === "string" && payload.last_name.trim()
      ? payload.last_name.trim()
      : null;
  const mergedFromParts = [firstName, lastName].filter(Boolean).join(" ").trim() || null;
  const fullName =
    typeof payload.full_name === "string" && payload.full_name.trim()
      ? payload.full_name.trim()
      : mergedFromParts;
  const driverName =
    typeof payload.driver_name === "string" && payload.driver_name.trim()
      ? payload.driver_name.trim()
      : fullName;
  const missionStatus =
    typeof payload.mission_status === "string" ? payload.mission_status.trim().toUpperCase() : "";
  const driverStatus =
    typeof payload.status === "string" ? payload.status.trim().toLowerCase() : "";
  const noActiveMission =
    missionStatus === "NONE" || missionStatus === "" || driverStatus === "available";
  const rawMissionId = payload.mission_id;
  const missionId =
    noActiveMission
      ? null
      : rawMissionId != null && Number.isFinite(Number(rawMissionId))
        ? Number(rawMissionId)
        : null;

  return {
    driver_id: driverId,
    driver_name: driverName,
    full_name: fullName ?? driverName,
    first_name: firstName,
    last_name: lastName,
    mission_id: missionId,
    latitude,
    longitude,
    timestamp: payload.timestamp ?? new Date().toISOString(),
    accuracy: payload.accuracy ?? null,
    heading: payload.heading ?? null,
    speed: payload.speed ?? null,
    is_background: payload.is_background,
    recorded_at: payload.recorded_at ?? payload.timestamp ?? null,
    received_at: payload.received_at ?? new Date().toISOString(),
    last_seen_seconds: lastSeenSeconds,
    location_status: normalizeLocationStatus(payload.location_status ?? null, lastSeenSeconds),
    status: driverStatus || undefined,
    presence_status:
      typeof payload.presence_status === "string" ? payload.presence_status.trim().toLowerCase() : undefined,
    tracking_display_status:
      typeof payload.tracking_display_status === "string"
        ? payload.tracking_display_status.trim().toLowerCase()
        : undefined,
    device_health:
      payload.device_health && typeof payload.device_health === "object"
        ? {
            constraint_reason:
              typeof payload.device_health.constraint_reason === "string"
                ? payload.device_health.constraint_reason
                : payload.device_health.constraint_reason ?? null,
            battery_optimized: payload.device_health.battery_optimized ?? null,
            tracking_active: payload.device_health.tracking_active ?? null,
          }
        : undefined,
  };
}

export function shouldReplaceDriverLocation(
  current: CompanyDriverLiveLocation | undefined,
  incoming: CompanyDriverLiveLocation
): boolean {
  if (!current) return true;

  // Une position marquée observability_only ne doit jamais écraser une position
  // live plus récente. Aligné sur la logique ops driverLiveMerge.ts.
  if (incoming.accepted_observability_only && !current.accepted_observability_only) {
    const incomingTs = toEpoch(incoming.recorded_at ?? incoming.timestamp);
    const currentTs = toEpoch(current.recorded_at ?? current.timestamp);
    if (incomingTs <= currentTs) return false;
  }

  const incomingRecorded = toEpoch(incoming.recorded_at ?? null);
  const currentRecorded = toEpoch(current.recorded_at ?? null);
  if (incomingRecorded !== currentRecorded) return incomingRecorded > currentRecorded;

  const incomingTimestamp = toEpoch(incoming.timestamp);
  const currentTimestamp = toEpoch(current.timestamp);
  if (incomingTimestamp !== currentTimestamp) return incomingTimestamp > currentTimestamp;

  const incomingReceived = toEpoch(incoming.received_at ?? null);
  const currentReceived = toEpoch(current.received_at ?? null);
  return incomingReceived > currentReceived;
}

function mergeRealtimeDriver(
  existing: CompanyDriverLiveLocation | undefined,
  normalized: CompanyDriverLiveLocation
): CompanyDriverLiveLocation | null {
  const merged: CompanyDriverLiveLocation = {
    ...normalized,
    driver_name: normalized.driver_name ?? existing?.driver_name ?? null,
    full_name: normalized.full_name ?? existing?.full_name ?? null,
    first_name: normalized.first_name ?? existing?.first_name ?? null,
    last_name: normalized.last_name ?? existing?.last_name ?? null,
    status: normalized.status ?? existing?.status ?? undefined,
    presence_status: normalized.presence_status ?? existing?.presence_status ?? undefined,
    tracking_display_status:
      normalized.tracking_display_status ?? existing?.tracking_display_status ?? undefined,
    device_health: normalized.device_health ?? existing?.device_health ?? undefined,
  };
  if (!shouldReplaceDriverLocation(existing, normalized)) return null;
  const preserved = mergeDriverLocationPreserveReference(existing, merged);
  if (preserved !== existing) return preserved;
  // Parité web : appliquer les micro-mouvements GPS même sous le seuil métier (5 m).
  return {
    ...existing,
    latitude: merged.latitude,
    longitude: merged.longitude,
    recorded_at: merged.recorded_at ?? existing.recorded_at,
    received_at: merged.received_at ?? existing.received_at,
    timestamp: merged.timestamp,
    heading: merged.heading ?? existing.heading,
    speed: merged.speed ?? existing.speed,
    accuracy: merged.accuracy ?? existing.accuracy,
    location_status: merged.location_status ?? existing.location_status,
    tracking_display_status:
      merged.tracking_display_status ?? existing.tracking_display_status,
    last_seen_seconds: merged.last_seen_seconds ?? existing.last_seen_seconds,
    status: merged.status ?? existing.status,
    mission_id: merged.mission_id ?? existing.mission_id,
    is_background: merged.is_background ?? existing.is_background,
  };
}

export function applyPendingDriverUpdates(
  currentMap: Record<number, CompanyDriverLiveLocation>,
  pending: Map<number, CompanyDriverLiveLocation>
): Record<number, CompanyDriverLiveLocation> {
  if (pending.size === 0) return currentMap;

  let changed = false;
  const nextMap = { ...currentMap };

  pending.forEach((incoming, driverId) => {
    const existing = currentMap[driverId];
    const resolved = mergeRealtimeDriver(existing, incoming);
    if (resolved == null) return;
    if (resolved === existing) return;
    nextMap[driverId] = resolved;
    changed = true;
  });

  return changed ? nextMap : currentMap;
}

export function useCompanyDriverLiveTracking() {
  const contextId = useActiveCompanyContextId();
  const snapshotQuery = useCompanyDriversLocationsSnapshotQuery();
  const refetchSnapshot = snapshotQuery.refetch;
  const [driversMap, setDriversMap] = useState<Record<number, CompanyDriverLiveLocation>>({});
  const lastRealtimeEventAtRef = useRef<number>(0);
  const pendingRef = useRef<Map<number, CompanyDriverLiveLocation>>(new Map());
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const maxAgeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const batchStartedAtRef = useRef<number | null>(null);
  const flushInProgressRef = useRef(false);
  const flushAgainRef = useRef(false);
  const firstGpsAfterReconnectRef = useRef(false);
  const driversMapRef = useRef(driversMap);
  driversMapRef.current = driversMap;

  useEffect(() => {
    configureGpsMetricsSampling(5);
  }, []);

  const flushPending = useCallback(() => {
    if (pendingRef.current.size === 0) {
      batchStartedAtRef.current = null;
      if (maxAgeTimerRef.current != null) {
        clearTimeout(maxAgeTimerRef.current);
        maxAgeTimerRef.current = null;
      }
      return;
    }

    if (flushInProgressRef.current) {
      flushAgainRef.current = true;
      recordGpsBatchCoalesced(pendingRef.current.size);
      return;
    }

    flushInProgressRef.current = true;
    const batch = pendingRef.current;
    pendingRef.current = new Map();
    batchStartedAtRef.current = null;
    if (maxAgeTimerRef.current != null) {
      clearTimeout(maxAgeTimerRef.current);
      maxAgeTimerRef.current = null;
    }

    setDriversMap((currentMap) => {
      const next = applyPendingDriverUpdates(currentMap, batch);
      if (next !== currentMap && shouldSampleGpsMetricEvent()) {
        recordMapSetState();
      }
      return next;
    });
    recordGpsBatchFlush();

    flushInProgressRef.current = false;
    if (flushAgainRef.current) {
      flushAgainRef.current = false;
      flushPending();
    }
  }, []);

  const scheduleMaxAgeFlush = useCallback(() => {
    if (maxAgeTimerRef.current != null || batchStartedAtRef.current == null) return;
    const elapsed = Date.now() - batchStartedAtRef.current;
    const remaining = Math.max(0, getEffectiveMaxBatchAgeMs() - elapsed);
    maxAgeTimerRef.current = setTimeout(() => {
      maxAgeTimerRef.current = null;
      flushPending();
    }, remaining);
  }, [flushPending]);

  const scheduleFlushForPriority = useCallback(
    (priority: GpsFlushPriority) => {
      if (flushInProgressRef.current) {
        flushAgainRef.current = true;
        return;
      }
      scheduleMaxAgeFlush();
      const lanesEnabled = isFeatureEnabled("company_gps_flush_priority_lanes_enabled");
      const delayMs = resolveFlushDelayMs(priority, lanesEnabled);
      if (delayMs <= 0) {
        if (flushTimerRef.current != null) {
          clearTimeout(flushTimerRef.current);
          flushTimerRef.current = null;
        }
        flushPending();
        return;
      }
      if (flushTimerRef.current != null) return;
      flushTimerRef.current = setTimeout(() => {
        flushTimerRef.current = null;
        flushPending();
      }, delayMs);
    },
    [flushPending, scheduleMaxAgeFlush]
  );

  const enqueueDriverUpdate = useCallback(
    (normalized: CompanyDriverLiveLocation, options?: {
      immediate?: boolean;
      priority?: GpsFlushPriority;
      observabilityOnly?: boolean;
    }) => {
      const existing = driversMapRef.current[normalized.driver_id];
      const resolved = mergeRealtimeDriver(existing, normalized);
      if (resolved == null || resolved === existing) return;

      if (batchStartedAtRef.current == null) {
        batchStartedAtRef.current = Date.now();
      }
      pendingRef.current.set(normalized.driver_id, resolved);

      const priority =
        options?.priority ??
        (options?.immediate ? "critical" : "visible");

      if (options?.immediate || priority === "critical") {
        if (flushTimerRef.current != null) {
          clearTimeout(flushTimerRef.current);
          flushTimerRef.current = null;
        }
        flushPending();
        return;
      }

      scheduleFlushForPriority(priority);
    },
    [flushPending, scheduleFlushForPriority]
  );

  useEffect(() => {
    const snapshotLocations = snapshotQuery.data?.locations ?? [];
    if (snapshotLocations.length === 0) return;
    setDriversMap((currentMap) => {
      let changed = false;
      const nextMap = { ...currentMap };
      snapshotLocations.forEach((driver) => {
        const currentDriver = nextMap[driver.driver_id];
        const normalizedDriver: CompanyDriverLiveLocation = {
          ...driver,
          recorded_at: driver.recorded_at ?? driver.timestamp,
          received_at: driver.received_at ?? snapshotQuery.data?.refreshed_at ?? new Date().toISOString(),
        };
        if (!shouldReplaceDriverLocation(currentDriver, normalizedDriver)) return;
        const resolved = mergeDriverLocationPreserveReference(currentDriver, normalizedDriver);
        if (resolved === currentDriver) return;
        nextMap[driver.driver_id] = resolved;
        changed = true;
      });
      return changed ? nextMap : currentMap;
    });
  }, [snapshotQuery.data]);

  useEffect(() => {
    if (!contextId) return;
    return contextRealtimeRouter.subscribe(contextId, (event) => {
      if (!event || typeof event !== "object") return;
      const payload = event as {
        event_type?: string;
      } & DriverRealtimePayload;
      if (
        payload.event_type !== "driver_location_update" &&
        payload.event_type !== "driver_live_state_update" &&
        payload.event_type !== "company_socket_reconnected"
      ) {
        return;
      }
      if (payload.event_type === "company_socket_reconnected") {
        firstGpsAfterReconnectRef.current = true;
        if (pendingRef.current.size > 0) {
          flushPending();
        }
        void refetchSnapshot();
        return;
      }

      const isCriticalStateEvent = payload.event_type === "driver_live_state_update";
      lastRealtimeEventAtRef.current = Date.now();

      if (isCriticalStateEvent && pendingRef.current.size > 0) {
        flushPending();
      }

      const normalized = normalizeRealtimeLocation(payload);
      if (!normalized) return;

      const isLocationUpdate = payload.event_type === "driver_location_update";
      const immediate =
        isCriticalStateEvent ||
        isLocationUpdate ||
        (firstGpsAfterReconnectRef.current && isLocationUpdate);

      if (firstGpsAfterReconnectRef.current && isLocationUpdate) {
        firstGpsAfterReconnectRef.current = false;
      }

      const priority: GpsFlushPriority = isLocationUpdate
        ? "critical"
        : resolveGpsEventFlushPriority(payload.event_type ?? "", {
            immediate,
            observabilityOnly: Boolean(normalized.accepted_observability_only),
          });

      enqueueDriverUpdate(normalized, { immediate, priority });
    });
  }, [contextId, enqueueDriverUpdate, flushPending, refetchSnapshot]);

  useEffect(() => {
    if (!contextId) return;
    const interval = setInterval(() => {
      if (lastRealtimeEventAtRef.current === 0) return;
      if (Date.now() - lastRealtimeEventAtRef.current < MAP_SILENCE_RESYNC_MS) return;
      void refetchSnapshot();
      lastRealtimeEventAtRef.current = Date.now();
    }, 10_000);
    return () => clearInterval(interval);
  }, [contextId, refetchSnapshot]);

  useEffect(() => {
    return () => {
      if (flushTimerRef.current != null) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      if (maxAgeTimerRef.current != null) {
        clearTimeout(maxAgeTimerRef.current);
        maxAgeTimerRef.current = null;
      }
      if (pendingRef.current.size > 0) {
        const batch = pendingRef.current;
        pendingRef.current = new Map();
        setDriversMap((currentMap) => applyPendingDriverUpdates(currentMap, batch));
      }
    };
  }, []);

  const sortedDriversRef = useRef<CompanyDriverLiveLocation[]>([]);
  const drivers = useMemo(() => {
    const next = Object.values(driversMap).sort((a, b) => a.driver_id - b.driver_id);
    const prev = sortedDriversRef.current;
    if (
      prev.length === next.length &&
      prev.every((driver, index) => driver === next[index])
    ) {
      return prev;
    }
    sortedDriversRef.current = next;
    return next;
  }, [driversMap]);

  return {
    drivers,
    isLoading: snapshotQuery.isLoading,
    error: snapshotQuery.error,
    refetch: snapshotQuery.refetch,
    snapshotRefreshedAt: snapshotQuery.data?.refreshed_at ?? null,
  };
}
