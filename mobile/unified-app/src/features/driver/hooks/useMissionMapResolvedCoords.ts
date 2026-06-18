import { useEffect, useMemo, useState } from "react";
import { geocodeMissionAddress } from "../domain/geocodeMissionAddress";
import {
  missionCoordToMapPoint,
  resolveStaticMissionMapCoords,
  type MapLatLng,
  type MissionMapCoordInput,
} from "../domain/missionMapCoordUtils";

export function useMissionMapResolvedCoords(input: MissionMapCoordInput) {
  const staticCoords = useMemo(
    () => resolveStaticMissionMapCoords(input),
    [input]
  );
  const [geocodedPickup, setGeocodedPickup] = useState<MapLatLng | null>(null);
  const [geocodedDropoff, setGeocodedDropoff] = useState<MapLatLng | null>(null);
  const [resolving, setResolving] = useState(false);

  const pickupAddress = input.pickupLocation?.trim() ?? "";
  const dropoffAddress = input.dropoffLocation?.trim() ?? "";

  useEffect(() => {
    setGeocodedPickup(null);
    setGeocodedDropoff(null);

    const needsPickup = !staticCoords.pickupCoord && pickupAddress.length > 0;
    const needsDropoff = !staticCoords.dropoffCoord && dropoffAddress.length > 0;
    if (!needsPickup && !needsDropoff) {
      setResolving(false);
      return;
    }

    let cancelled = false;
    setResolving(true);

    void (async () => {
      try {
        const [pickupResult, dropoffResult] = await Promise.all([
          needsPickup ? geocodeMissionAddress(pickupAddress) : Promise.resolve(null),
          needsDropoff ? geocodeMissionAddress(dropoffAddress) : Promise.resolve(null),
        ]);
        if (cancelled) return;
        setGeocodedPickup(missionCoordToMapPoint(pickupResult));
        setGeocodedDropoff(missionCoordToMapPoint(dropoffResult));
      } finally {
        if (!cancelled) setResolving(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [
    staticCoords.pickupCoord,
    staticCoords.dropoffCoord,
    pickupAddress,
    dropoffAddress,
  ]);

  const pickupCoord = staticCoords.pickupCoord ?? geocodedPickup;
  const dropoffCoord = staticCoords.dropoffCoord ?? geocodedDropoff;
  const driverCoord = staticCoords.driverCoord;
  const fallbackCoord = driverCoord ?? pickupCoord ?? dropoffCoord;

  return {
    driverCoord,
    pickupCoord,
    dropoffCoord,
    fallbackCoord,
    resolving: resolving && !fallbackCoord,
  };
}
