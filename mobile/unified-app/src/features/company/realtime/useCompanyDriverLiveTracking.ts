import { useEffect, useMemo, useRef, useState } from "react";
import { contextRealtimeRouter } from "../../../core/realtime/contextRealtimeRouter";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import { useActiveCompanyContextId, useCompanyDriversLocationsSnapshotQuery } from "../hooks";

export const MAP_SILENCE_RESYNC_MS = 120_000;
const STALE_SECONDS_THRESHOLD = 120;

function toEpoch(value: string | null | undefined): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

type DriverRealtimePayload = Partial<CompanyDriverLiveLocation> & {
  driver_id?: number | string;
  lat?: number;
  lon?: number;
  lng?: number;
  last_seen_seconds?: number | null;
  location_status?: "live" | "recent" | "stale" | "offline" | null;
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
  return {
    driver_id: driverId,
    mission_id: payload.mission_id ?? null,
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

export function useCompanyDriverLiveTracking() {
  const contextId = useActiveCompanyContextId();
  const snapshotQuery = useCompanyDriversLocationsSnapshotQuery();
  const refetchSnapshot = snapshotQuery.refetch;
  const [driversMap, setDriversMap] = useState<Record<number, CompanyDriverLiveLocation>>({});
  const lastRealtimeEventAtRef = useRef<number>(0);

  useEffect(() => {
    const snapshotLocations = snapshotQuery.data?.locations ?? [];
    if (snapshotLocations.length === 0) return;
    setDriversMap((currentMap) => {
      const nextMap = { ...currentMap };
      snapshotLocations.forEach((driver) => {
        const currentDriver = nextMap[driver.driver_id];
        const normalizedDriver: CompanyDriverLiveLocation = {
          ...driver,
          recorded_at: driver.recorded_at ?? driver.timestamp,
          received_at: driver.received_at ?? snapshotQuery.data?.refreshed_at ?? new Date().toISOString(),
        };
        if (shouldReplaceDriverLocation(currentDriver, normalizedDriver)) {
          nextMap[driver.driver_id] = normalizedDriver;
        }
      });
      return nextMap;
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
        void refetchSnapshot();
        return;
      }
      const normalized = normalizeRealtimeLocation(payload);
      if (!normalized) return;
      lastRealtimeEventAtRef.current = Date.now();
      setDriversMap((currentMap) => {
        const existing = currentMap[normalized.driver_id];
        if (!shouldReplaceDriverLocation(existing, normalized)) {
          return currentMap;
        }
        return {
          ...currentMap,
          [normalized.driver_id]: normalized,
        };
      });
    });
  }, [contextId, refetchSnapshot]);

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

  const drivers = useMemo(
    () => Object.values(driversMap).sort((a, b) => a.driver_id - b.driver_id),
    [driversMap]
  );

  return {
    drivers,
    isLoading: snapshotQuery.isLoading,
    error: snapshotQuery.error,
    refetch: snapshotQuery.refetch,
    snapshotRefreshedAt: snapshotQuery.data?.refreshed_at ?? null,
  };
}
