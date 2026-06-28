import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { Marker } from "react-native-maps";

import type { CompanyDriverLiveLocation } from "../../api/contracts";
import {
  interpolateFleetMarkerPosition,
  isValidFleetMapCoordinate,
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
}: Options) {
  const [displayCoordinate, setDisplayCoordinate] = useState<FleetMapLatLng>(target);
  const displayCoordinateRef = useRef<FleetMapLatLng>(target);
  const committedCoordinateRef = useRef<FleetMapLatLng>(target);
  const previousMarkerKeyRef = useRef(markerKey);
  const previousRecordedAtMsRef = useRef<number | null>(parseFleetMarkerRecordedAtMs(recordedAt));
  const lastMotionAtMsRef = useRef<number | null>(null);
  const primaryMarkerRef = useRef<Marker | null>(null);
  const animationSeqRef = useRef(0);
  const rafCleanupRef = useRef<(() => void) | null>(null);

  const clearRaf = useCallback(() => {
    rafCleanupRef.current?.();
    rafCleanupRef.current = null;
  }, []);

  const snapTo = useCallback(
    (coordinate: FleetMapLatLng, seq: number) => {
      if (!shouldApplyFleetMarkerCommit(seq, animationSeqRef.current)) return;
      clearRaf();
      committedCoordinateRef.current = coordinate;
      displayCoordinateRef.current = coordinate;
      setDisplayCoordinate(coordinate);
      lastMotionAtMsRef.current = Date.now();
    },
    [clearRaf]
  );

  const runJsInterpolation = useCallback(
    (from: FleetMapLatLng, to: FleetMapLatLng, durationMs: number, seq: number) => {
      clearRaf();
      const startMs = performance.now();
      let rafId = 0;

      const step = () => {
        if (!shouldApplyFleetMarkerCommit(seq, animationSeqRef.current)) return;
        const elapsed = performance.now() - startMs;
        const progress = durationMs <= 0 ? 1 : Math.min(1, elapsed / durationMs);
        const coord = interpolateFleetMarkerPosition(from, to, progress);
        setDisplayCoordinate(coord);
        if (progress < 1) {
          rafId = requestAnimationFrame(step);
        } else {
          committedCoordinateRef.current = to;
          lastMotionAtMsRef.current = Date.now();
          rafCleanupRef.current = null;
        }
      };

      rafId = requestAnimationFrame(step);
      rafCleanupRef.current = () => cancelAnimationFrame(rafId);
    },
    [clearRaf]
  );

  useEffect(() => {
    return () => {
      animationSeqRef.current += 1;
      clearRaf();
    };
  }, [clearRaf]);

  useEffect(() => {
    const from = displayCoordinateRef.current;
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
      lastMotionAtMs: lastMotionAtMsRef.current,
      locationStatus,
      markerKeyChanged,
    });

    previousRecordedAtMsRef.current = nextRecordedAtMs ?? previousRecordedAtMsRef.current;

    animationSeqRef.current += 1;
    const seq = animationSeqRef.current;

    if (plan.mode === "snap") {
      snapTo(target, seq);
      return;
    }

    if (!isValidFleetMapCoordinate(from)) {
      snapTo(target, seq);
      return;
    }

    runJsInterpolation(from, target, plan.durationMs, seq);
  }, [
    locationStatus,
    markerKey,
    recordedAt,
    runJsInterpolation,
    snapTo,
    target.latitude,
    target.longitude,
  ]);

  return { displayCoordinate, primaryMarkerRef };
}
