import { useEffect, useMemo, useState } from "react";
import type { DriverLocationAckStatus } from "../types";
import { getTrackingSnapshot, subscribeTrackingSnapshot } from "../tracking";

export type TrackingRuntimeMode =
  | "mission_live"
  | "availability_presence"
  | "idle";

export type TrackingState = {
  isTracking: boolean;
  mode: TrackingRuntimeMode;
  fsmState?: string;
  lastUpdate?: number;
  lastAckAt?: number;
  lastAckIsQueued?: boolean;
  lastAckStatus?: DriverLocationAckStatus | null;
  lastAckError?: string | null;
  currentAttemptSeq: number;
  lastAckAttemptSeq?: number | null;
  currentAttemptEventId?: string | null;
  lastAckEventId?: string | null;
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
    fsmState: snapshot.fsmState,
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
    lastAckStatus: snapshot.lastAckStatus,
    lastAckError: snapshot.lastAckError,
    currentAttemptSeq: snapshot.currentAttemptSeq,
    lastAckAttemptSeq: snapshot.lastAckAttemptSeq,
    currentAttemptEventId: snapshot.currentAttemptEventId,
    lastAckEventId: snapshot.lastAckEventId,
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
    a.lastAckStatus === b.lastAckStatus &&
    a.lastAckError === b.lastAckError &&
    a.currentAttemptSeq === b.currentAttemptSeq &&
    a.lastAckAttemptSeq === b.lastAckAttemptSeq &&
    a.currentAttemptEventId === b.currentAttemptEventId &&
    a.lastAckEventId === b.lastAckEventId &&
    a.queueDepth === b.queueDepth &&
    a.missionId === b.missionId &&
    a.fsmState === b.fsmState
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
