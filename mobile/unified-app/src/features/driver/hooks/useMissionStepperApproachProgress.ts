import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DriverEtaSnapshot } from "../api";
import type { DriverMission } from "../types";
import { extractMissionMapCoordInput } from "../domain/missionMapCoordUtils";
import {
  resolveActiveStepperApproach,
  resolveDriverOriginForStepperApproach,
  resolveStepperApproachDistance,
  resolveStepperApproachProgressForMission,
  type ActiveStepperApproach,
} from "../domain/missionStepperApproach";
import { mapLatLngToMissionCoord } from "../domain/missionRouteMetrics";
import {
  getDriverLastKnownPosition,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import { useMissionMapResolvedCoords } from "./useMissionMapResolvedCoords";

const REFRESH_MS = 4_000;

export type MissionStepperApproachState = ActiveStepperApproach & {
  progress: number;
};

/**
 * Progression 0→1 sur le segment actif du stepper :
 * EN ROUTE → pickup, EN COURSE → destination.
 */
export function useMissionStepperApproachProgress(
  mission: DriverMission,
  options?: { etaSnapshot?: DriverEtaSnapshot | null; remainingDistanceKm?: number | null }
) {
  const etaSnapshot = options?.etaSnapshot;
  const activeApproach = useMemo(() => resolveActiveStepperApproach(mission), [mission]);
  const enabled = activeApproach != null;

  const mapInput = useMemo(() => extractMissionMapCoordInput(mission), [mission]);
  const { pickupCoord, dropoffCoord } = useMissionMapResolvedCoords(mapInput);
  const pickup = useMemo(() => mapLatLngToMissionCoord(pickupCoord), [pickupCoord]);
  const dropoff = useMemo(() => mapLatLngToMissionCoord(dropoffCoord), [dropoffCoord]);

  const [trackingPosition, setTrackingPosition] = useState(() =>
    enabled ? getDriverLastKnownPosition() : null
  );
  const displayedRef = useRef(0);
  const [approachState, setApproachState] = useState<MissionStepperApproachState | null>(null);

  const measureProgress = useCallback((): number | null => {
    if (!activeApproach || typeof mission.id !== "number") return null;

    const target =
      activeApproach.segment === "pickup" ? pickup : dropoff;
    const driver = resolveDriverOriginForStepperApproach(mission, trackingPosition, etaSnapshot);
    const routedMeters =
      options?.remainingDistanceKm != null && options.remainingDistanceKm > 0
        ? options.remainingDistanceKm * 1000
        : null;
    const currentMeters = routedMeters ?? resolveStepperApproachDistance(driver, target);
    if (currentMeters == null) return null;

    return resolveStepperApproachProgressForMission(
      mission.id,
      activeApproach.segment,
      currentMeters
    );
  }, [
    activeApproach,
    mission,
    trackingPosition,
    pickup,
    dropoff,
    etaSnapshot,
    options?.remainingDistanceKm,
  ]);

  useEffect(() => {
    if (!enabled) {
      setTrackingPosition(null);
      displayedRef.current = 0;
      setApproachState(null);
      return;
    }
    setTrackingPosition(getDriverLastKnownPosition());
    return subscribeDriverTrackingBridge((snapshot) => {
      setTrackingPosition(snapshot.lastPosition);
    });
  }, [enabled, mission.id]);

  useEffect(() => {
    if (!enabled || !activeApproach) return;

    const apply = () => {
      const target = measureProgress();
      if (target == null) return;
      const next = Math.max(displayedRef.current, target);
      if (Math.abs(next - displayedRef.current) < 0.002) return;
      displayedRef.current = next;
      setApproachState({ ...activeApproach, progress: next });
    };

    apply();
    const interval = setInterval(apply, REFRESH_MS);
    return () => clearInterval(interval);
  }, [enabled, activeApproach, measureProgress]);

  return approachState;
}
