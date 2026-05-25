import { useIsFocused } from "@react-navigation/native";
import { useEffect, useMemo, useState } from "react";
import type { DriverEtaSnapshot } from "../api";
import type { DriverMission } from "../types";
import { fetchPickupDropoffRouteMetrics, fetchRouteMetrics } from "../domain/fetchRouteMetrics";
import {
  estimateMissionRouteMetricsBetweenCoords,
  extractMissionDropoffCoord,
  extractMissionPickupCoord,
  formatMissionRouteDistanceKm,
  formatMissionRouteDurationMinutes,
  mapLatLngToMissionCoord,
  resolveLiveDriverOrigin,
  resolveMissionRouteLeg,
  resolveMissionRouteMetrics,
  type MissionCoord,
  type MissionRouteMetrics,
} from "../domain/missionRouteMetrics";
import {
  extractMissionMapCoordInput,
} from "../domain/missionMapCoordUtils";
import { resolveDriverStatusForUx } from "../statusDictionary";
import {
  getDriverLastKnownPosition,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import { useMissionMapResolvedCoords } from "./useMissionMapResolvedCoords";

const LIVE_REFRESH_MS = 20_000;

function mergeDurationWithEta(
  metrics: MissionRouteMetrics,
  etaMinutes: number | null | undefined
): MissionRouteMetrics {
  if (metrics.durationMinutes != null && metrics.durationMinutes > 0) return metrics;
  if (etaMinutes == null || !Number.isFinite(etaMinutes) || etaMinutes <= 0) return metrics;
  return { ...metrics, durationMinutes: Math.round(etaMinutes) };
}

function resolveEtaMinutes(
  etaSnapshot: DriverEtaSnapshot | null | undefined,
  fallbackEtaMinutes: number | null | undefined
): number | null {
  const fromSnapshot = etaSnapshot?.eta_minutes;
  if (fromSnapshot != null && Number.isFinite(fromSnapshot) && fromSnapshot > 0) {
    return Math.round(fromSnapshot);
  }
  if (fallbackEtaMinutes != null && Number.isFinite(fallbackEtaMinutes) && fallbackEtaMinutes > 0) {
    return Math.round(fallbackEtaMinutes);
  }
  return null;
}

export function useMissionRouteMetrics(
  mission: DriverMission,
  options?: { etaMinutes?: number | null; etaSnapshot?: DriverEtaSnapshot | null }
) {
  const isFocused = useIsFocused();
  const statusKey = resolveDriverStatusForUx(mission.status);
  const leg = useMemo(() => resolveMissionRouteLeg(statusKey), [statusKey]);
  const staticMetrics = useMemo(() => resolveMissionRouteMetrics(mission), [mission]);
  const [liveMetrics, setLiveMetrics] = useState<MissionRouteMetrics | null>(null);
  const [trackingPosition, setTrackingPosition] = useState<MissionCoord | null>(() =>
    leg.mode === "live" ? getDriverLastKnownPosition() : null
  );

  const mapInput = useMemo(() => extractMissionMapCoordInput(mission), [mission]);
  const { pickupCoord, dropoffCoord, driverCoord: missionDriverCoord } =
    useMissionMapResolvedCoords(mapInput);

  const pickup = useMemo(
    () => mapLatLngToMissionCoord(pickupCoord) ?? extractMissionPickupCoord(mission),
    [pickupCoord, mission]
  );
  const dropoff = useMemo(
    () => mapLatLngToMissionCoord(dropoffCoord) ?? extractMissionDropoffCoord(mission),
    [dropoffCoord, mission]
  );
  const missionDriver = useMemo(
    () => mapLatLngToMissionCoord(missionDriverCoord),
    [missionDriverCoord]
  );

  const pickupAddress =
    typeof mission.pickup_location === "string" ? mission.pickup_location : null;
  const dropoffAddress =
    typeof mission.dropoff_location === "string" ? mission.dropoff_location : null;

  const etaMinutes = resolveEtaMinutes(options?.etaSnapshot, options?.etaMinutes);

  useEffect(() => {
    if (leg.mode !== "live") {
      setTrackingPosition(null);
      return;
    }
    setTrackingPosition(getDriverLastKnownPosition());
    return subscribeDriverTrackingBridge((snapshot) => {
      setTrackingPosition(snapshot.lastPosition);
    });
  }, [leg.mode, mission.id]);

  useEffect(() => {
    if (!isFocused) {
      setLiveMetrics(null);
      return;
    }
    if (leg.mode === "live") {
      const destinationCoord = leg.destination === "pickup" ? pickup : dropoff;
      const destinationAddress = leg.destination === "pickup" ? pickupAddress : dropoffAddress;
      const hasDestination = Boolean(destinationCoord || destinationAddress?.trim());
      if (!hasDestination) {
        setLiveMetrics(null);
        return;
      }

      let cancelled = false;
      const refresh = async () => {
        const origin = resolveLiveDriverOrigin({
          tracking: trackingPosition ?? getDriverLastKnownPosition(),
          etaDriver: options?.etaSnapshot
            ? {
                lat: options.etaSnapshot.driver_lat ?? null,
                lon: options.etaSnapshot.driver_lon ?? null,
              }
            : null,
          mission: missionDriver,
        });

        if (!origin) {
          if (!cancelled) setLiveMetrics(null);
          return;
        }

        const routed = await fetchRouteMetrics({
          origin,
          destination: destinationCoord,
          destinationAddress: destinationCoord ? null : destinationAddress,
        });

        if (cancelled) return;

        if (routed) {
          setLiveMetrics(mergeDurationWithEta(routed, etaMinutes));
          return;
        }

        if (destinationCoord) {
          setLiveMetrics(
            mergeDurationWithEta(
              estimateMissionRouteMetricsBetweenCoords(origin, destinationCoord),
              etaMinutes
            )
          );
          return;
        }

        setLiveMetrics(
          etaMinutes != null ? { distanceKm: null, durationMinutes: etaMinutes } : null
        );
      };

      void refresh();
      const interval = setInterval(() => {
        void refresh();
      }, LIVE_REFRESH_MS);

      return () => {
        cancelled = true;
        clearInterval(interval);
      };
    }

    const hasStaticDistance =
      staticMetrics.distanceKm != null && staticMetrics.distanceKm > 0;
    const hasStaticDuration =
      staticMetrics.durationMinutes != null && staticMetrics.durationMinutes > 0;
    if (hasStaticDistance && hasStaticDuration) {
      setLiveMetrics(null);
      return;
    }

    const canFetchCoords = Boolean(pickup && dropoff);
    const canFetchAddresses = Boolean(pickupAddress?.trim() && dropoffAddress?.trim());
    if (!canFetchCoords && !canFetchAddresses) {
      setLiveMetrics(null);
      return;
    }

    let cancelled = false;
    void fetchPickupDropoffRouteMetrics({
      pickup,
      dropoff,
      pickupAddress,
      dropoffAddress,
    }).then((result) => {
      if (cancelled || !result) return;
      setLiveMetrics(result);
    });

    return () => {
      cancelled = true;
    };
  }, [
    leg,
    trackingPosition,
    mission.id,
    pickup,
    dropoff,
    missionDriver,
    pickupAddress,
    dropoffAddress,
    staticMetrics.distanceKm,
    staticMetrics.durationMinutes,
    etaMinutes,
    options?.etaSnapshot?.driver_lat,
    options?.etaSnapshot?.driver_lon,
    isFocused,
  ]);

  const plannedMetrics = liveMetrics ?? staticMetrics;

  let distanceKm: number | null;
  let durationMinutes: number | null;

  if (leg.mode === "live") {
    distanceKm = liveMetrics?.distanceKm ?? null;
    durationMinutes =
      liveMetrics?.durationMinutes ?? etaMinutes ?? null;
  } else {
    distanceKm = plannedMetrics.distanceKm;
    durationMinutes = plannedMetrics.durationMinutes;
  }

  const distanceMetricLabel = leg.mode === "live" ? "DISTANCE RESTANTE" : "DISTANCE";

  const durationMetricLabel =
    leg.mode === "live"
      ? leg.destination === "pickup"
        ? "TEMPS PRISE EN CHARGE"
        : "TEMPS DESTINATION"
      : "DURÉE ESTIMÉE";

  return {
    distanceLabel: formatMissionRouteDistanceKm(distanceKm),
    durationLabel: formatMissionRouteDurationMinutes(durationMinutes),
    distanceKm,
    durationMinutes,
    distanceMetricLabel,
    durationMetricLabel,
    legMode: leg.mode,
    legDestination: leg.mode === "live" ? leg.destination : null,
  };
}
