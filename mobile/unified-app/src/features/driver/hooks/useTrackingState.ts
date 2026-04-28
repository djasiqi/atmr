import { useEffect, useMemo, useState } from "react";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { getTrackingSnapshot, subscribeTrackingSnapshot } from "../tracking";

export type TrackingRuntimeMode =
  | "mission_live"
  | "availability_presence"
  | "idle";

export type TrackingState = {
  isTracking: boolean;
  mode: TrackingRuntimeMode;
  lastUpdate?: number;
  accuracy?: number;
};

function mapSnapshotToTrackingState(
  snapshot: ReturnType<typeof getTrackingSnapshot>
): TrackingState {
  const mode: TrackingRuntimeMode = !snapshot.isRunning
    ? "idle"
    : snapshot.missionStatus === "ASSIGNED"
      ? "availability_presence"
      : "mission_live";
  return {
    isTracking: snapshot.isRunning,
    mode,
    lastUpdate: snapshot.lastSentAt
      ? Number.isFinite(Date.parse(snapshot.lastSentAt))
        ? Date.parse(snapshot.lastSentAt)
        : undefined
      : undefined,
    accuracy: undefined,
  };
}

export function useTrackingState(): TrackingState {
  const [snapshot, setSnapshot] = useState(() => getTrackingSnapshot());

  useEffect(() => {
    const unsubscribeTracking = subscribeTrackingSnapshot((nextSnapshot) => {
      setSnapshot(nextSnapshot);
    });
    const unsubscribeRealtime = realtimeManager.subscribe(() => {
      setSnapshot(getTrackingSnapshot());
    });
    return () => {
      unsubscribeTracking();
      unsubscribeRealtime();
    };
  }, []);

  return useMemo(() => mapSnapshotToTrackingState(snapshot), [snapshot]);
}

