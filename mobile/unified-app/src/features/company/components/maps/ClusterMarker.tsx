import { useMemo } from "react";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { buildFleetClusterMarkerImageSource } from "./fleetNativeMarkerImage";
import { FLEET_MAP_MARKER_DIMMED_OPACITY } from "./fleetMapTypes";
import { isValidMapCoord } from "./mapsIosNewArchSafeMode";
import { useFleetMarkerMotion } from "./useFleetMarkerMotion";

type Props = {
  markerKey: string;
  latitude: number;
  longitude: number;
  count: number;
  drivers: FleetDriverMapItem[];
  onPress?: () => void;
  dimmed?: boolean;
};

/** Disque cluster web coloré selon statut dominant. */
export function ClusterMarker({
  markerKey,
  latitude,
  longitude,
  count,
  drivers,
  onPress,
  dimmed = false,
}: Props) {
  const opacity = dimmed ? FLEET_MAP_MARKER_DIMMED_OPACITY : 1;
  const targetCoordinate = useMemo(
    () => ({ latitude, longitude }),
    [latitude, longitude]
  );

  const { displayCoordinate, primaryMarkerRef } = useFleetMarkerMotion({
    target: targetCoordinate,
    markerKey,
  });

  const iconSource = useMemo(
    () => buildFleetClusterMarkerImageSource(count, drivers),
    [count, drivers]
  );

  if (!iconSource.uri?.trim()) {
    return null;
  }

  if (!isValidMapCoord(displayCoordinate.latitude, displayCoordinate.longitude)) {
    return null;
  }

  return (
    <FleetMapRasterMarker
      ref={primaryMarkerRef}
      coordinate={displayCoordinate}
      imageSource={iconSource}
      anchor={{ x: 0.5, y: 0.5 }}
      onPress={onPress}
      zIndex={500}
      opacity={opacity}
    />
  );
}
