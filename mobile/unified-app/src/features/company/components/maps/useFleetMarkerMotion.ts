import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { Marker } from "react-native-maps";

import type { CompanyDriverLiveLocation } from "../../api/contracts";
import {
  animateFleetMarkerToCoordinate,
  parseFleetMarkerRecordedAtMs,
  resolveFleetMarkerMotionPlan,
  shouldApplyFleetMarkerCommit,
  type FleetMapLatLng,
} from "./fleetMapMarkerMotion";

type Options = {
  target: FleetMapLatLng;
  markerKey: string;
  recordedAt?: string | null;
  locationStatus?: CompanyDriverLiveLocation["location_status"] | null;
  secondaryMarkerRef?: RefObject<Marker | null>;
};

export function useFleetMarkerMotion({
  target,
  markerKey,
  recordedAt,
  locationStatus,
  secondaryMarkerRef,
}: Options) {
  const [displayCoordinate, setDisplayCoordinate] = useState<FleetMapLatLng>(target);
  const committedCoordinateRef = useRef<FleetMapLatLng>(target);
  const previousMarkerKeyRef = useRef(markerKey);
  const previousRecordedAtMsRef = useRef<number | null>(parseFleetMarkerRecordedAtMs(recordedAt));
  const primaryMarkerRef = useRef<Marker | null>(null);
  const commitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animationSeqRef = useRef(0);

  const clearCommitTimer = useCallback(() => {
    if (commitTimerRef.current != null) {
      clearTimeout(commitTimerRef.current);
      commitTimerRef.current = null;
    }
  }, []);

  const snapTo = useCallback(
    (coordinate: FleetMapLatLng, seq: number) => {
      if (!shouldApplyFleetMarkerCommit(seq, animationSeqRef.current)) return;
      committedCoordinateRef.current = coordinate;
      setDisplayCoordinate(coordinate);
    },
    []
  );

  const animateMarkers = useCallback(
    (coordinate: FleetMapLatLng, durationMs: number): boolean => {
      const primaryOk = animateFleetMarkerToCoordinate(primaryMarkerRef.current, coordinate, durationMs);
      if (secondaryMarkerRef?.current) {
        animateFleetMarkerToCoordinate(secondaryMarkerRef.current, coordinate, durationMs);
      }
      return primaryOk;
    },
    [secondaryMarkerRef]
  );

  useEffect(() => {
    return () => {
      clearCommitTimer();
      animationSeqRef.current += 1;
    };
  }, [clearCommitTimer]);

  useEffect(() => {
    const from = committedCoordinateRef.current;
    const markerKeyChanged = markerKey !== previousMarkerKeyRef.current;
    if (markerKeyChanged) {
      previousMarkerKeyRef.current = markerKey;
    }

    const nextRecordedAtMs = parseFleetMarkerRecordedAtMs(recordedAt);
    const plan = resolveFleetMarkerMotionPlan({
      from,
      to: target,
      previousRecordedAtMs: previousRecordedAtMsRef.current,
      nextRecordedAtMs,
      locationStatus,
      markerKeyChanged,
    });

    previousRecordedAtMsRef.current = nextRecordedAtMs ?? previousRecordedAtMsRef.current;

    clearCommitTimer();
    animationSeqRef.current += 1;
    const seq = animationSeqRef.current;

    if (plan.mode === "snap") {
      snapTo(target, seq);
      return;
    }

    const animated = animateMarkers(target, plan.durationMs);
    if (!animated) {
      snapTo(target, seq);
      return;
    }

    commitTimerRef.current = setTimeout(() => {
      commitTimerRef.current = null;
      if (!shouldApplyFleetMarkerCommit(seq, animationSeqRef.current)) return;
      committedCoordinateRef.current = target;
      setDisplayCoordinate(target);
    }, plan.durationMs);
  }, [
    animateMarkers,
    clearCommitTimer,
    locationStatus,
    markerKey,
    recordedAt,
    snapTo,
    target.latitude,
    target.longitude,
  ]);

  return { displayCoordinate, primaryMarkerRef };
}
