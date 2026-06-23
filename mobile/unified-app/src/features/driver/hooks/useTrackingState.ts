import { useEffect, useMemo, useState } from "react";
import { getTrackingSnapshot, subscribeTrackingSnapshot } from "../tracking";

export type TrackingRuntimeMode =
  | "mission_live"
  | "availability_presence"
  | "idle";

export type TrackingState = {
  isTracking: boolean;
  mode: TrackingRuntimeMode;
  lastUpdate?: number;
  lastAckAt?: number;
  lastAckIsQueued?: boolean;
  queueDepth: number;
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
    lastAckAt: snapshot.lastAckAt
      ? Number.isFinite(Date.parse(snapshot.lastAckAt))
        ? Date.parse(snapshot.lastAckAt)
        : undefined
      : undefined,
    lastAckIsQueued: snapshot.lastAckIsQueued === true,
    queueDepth: snapshot.queueDepth,
    accuracy: undefined,
  };
}

function trackingSnapshotsEqual(
  a: ReturnType<typeof getTrackingSnapshot>,
  b: ReturnType<typeof getTrackingSnapshot>
): boolean {
  return (
    a.isRunning === b.isRunning &&
    a.missionStatus === b.missionStatus &&
    a.lastSentAt === b.lastSentAt &&
    a.lastAckAt === b.lastAckAt &&
    a.lastAckIsQueued === b.lastAckIsQueued &&
    a.queueDepth === b.queueDepth &&
    a.missionId === b.missionId
  );
}

export function useTrackingState(): TrackingState {
  const [snapshot, setSnapshot] = useState(() => getTrackingSnapshot());

  useEffect(() => {
    return subscribeTrackingSnapshot((nextSnapshot) => {
      setSnapshot((prev) => (trackingSnapshotsEqual(prev, nextSnapshot) ? prev : nextSnapshot));
    });
  }, []);

  return useMemo(() => mapSnapshotToTrackingState(snapshot), [snapshot]);
}

