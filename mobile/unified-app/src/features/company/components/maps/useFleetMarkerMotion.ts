import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import type { Marker } from "react-native-maps";

import { reportFleetMarkerAnimationSkipped } from "../../../../core/observability/fleetMapDiagnostics";
import type { CompanyDriverLiveLocation } from "../../api/contracts";
import {
  animateFleetMarkerToCoordinate,
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

function scheduleNativeMarkerReady(onReady: () => void): () => void {
  let cancelled = false;
  const frame1 = requestAnimationFrame(() => {
    if (cancelled) return;
    requestAnimationFrame(() => {
      if (!cancelled) {
        onReady();
      }
    });
  });
  return () => {
    cancelled = true;
    cancelAnimationFrame(frame1);
  };
}

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
  const primaryNativeReadyRef = useRef(false);
  const secondaryNativeReadyRef = useRef(false);
  const nativeReadyCleanupRef = useRef<(() => void) | null>(null);

  const markPrimaryNativePending = useCallback(() => {
    primaryNativeReadyRef.current = false;
    nativeReadyCleanupRef.current?.();
    nativeReadyCleanupRef.current = scheduleNativeMarkerReady(() => {
      primaryNativeReadyRef.current = true;
      nativeReadyCleanupRef.current = null;
    });
  }, []);

  const markSecondaryNativePending = useCallback(() => {
    secondaryNativeReadyRef.current = false;
    if (!secondaryMarkerRef?.current) {
      return;
    }
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (secondaryMarkerRef.current) {
          secondaryNativeReadyRef.current = true;
        }
      });
    });
  }, [secondaryMarkerRef]);

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
      markPrimaryNativePending();
      markSecondaryNativePending();
    },
    [markPrimaryNativePending, markSecondaryNativePending]
  );

  const animateMarkers = useCallback(
    (from: FleetMapLatLng, to: FleetMapLatLng, durationMs: number): boolean => {
      if (!primaryNativeReadyRef.current) {
        reportFleetMarkerAnimationSkipped("native_not_ready", { markerKey });
        return false;
      }

      const primaryOk = animateFleetMarkerToCoordinate({
        marker: primaryMarkerRef.current,
        from,
        to,
        durationMs,
      });

      if (
        secondaryMarkerRef?.current &&
        secondaryNativeReadyRef.current &&
        isValidFleetMapCoordinate(from) &&
        isValidFleetMapCoordinate(to)
      ) {
        animateFleetMarkerToCoordinate({
          marker: secondaryMarkerRef.current,
          from,
          to,
          durationMs,
          reportSkip: false,
        });
      }

      return primaryOk;
    },
    [markerKey, secondaryMarkerRef]
  );

  useEffect(() => {
    markPrimaryNativePending();
    return () => {
      clearCommitTimer();
      animationSeqRef.current += 1;
      nativeReadyCleanupRef.current?.();
      nativeReadyCleanupRef.current = null;
    };
  }, [clearCommitTimer, markPrimaryNativePending]);

  useEffect(() => {
    const from = committedCoordinateRef.current;
    const markerKeyChanged = markerKey !== previousMarkerKeyRef.current;
    if (markerKeyChanged) {
      previousMarkerKeyRef.current = markerKey;
      markPrimaryNativePending();
      markSecondaryNativePending();
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

    const animated = animateMarkers(from, target, plan.durationMs);
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
    markPrimaryNativePending,
    markSecondaryNativePending,
    markerKey,
    recordedAt,
    snapTo,
    target.latitude,
    target.longitude,
  ]);

  return { displayCoordinate, primaryMarkerRef };
}
