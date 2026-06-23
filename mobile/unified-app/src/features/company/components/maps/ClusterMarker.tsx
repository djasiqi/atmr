import { useMemo, useRef } from "react";
import { Marker } from "react-native-maps";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { ClusterCountBadgeMarker } from "./ClusterCountBadgeMarker";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { buildFleetDriverMarkerImageSource } from "./fleetNativeMarkerImage";
import { resolveFleetMarkerAnchor } from "./resolveFleetMarkerAnchor";
import { pickClusterRepresentativeStatus } from "./fleetLirieClusterMarker";
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
  /** Même transparence que les chauffeurs non sélectionnés. */
  dimmed?: boolean;
};

/** Icône PNG (`require`) + chiffre en Text natif (lisible sur Android). */
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
  const status = useMemo(() => pickClusterRepresentativeStatus(drivers), [drivers]);
  const targetCoordinate = useMemo(
    () => ({ latitude, longitude }),
    [latitude, longitude]
  );

  const badgeMarkerRef = useRef<Marker | null>(null);
  const { displayCoordinate, primaryMarkerRef } = useFleetMarkerMotion({
    target: targetCoordinate,
    markerKey,
    secondaryMarkerRef: badgeMarkerRef,
  });

  const iconSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, false),
    [status]
  );
  const markerAnchor = useMemo(
    () => resolveFleetMarkerAnchor(iconSource),
    [iconSource]
  );

  if (!iconSource.uri?.trim()) {
    return null;
  }

  if (!isValidMapCoord(displayCoordinate.latitude, displayCoordinate.longitude)) {
    return null;
  }

  return (
    <>
      <FleetMapRasterMarker
        ref={primaryMarkerRef}
        coordinate={displayCoordinate}
        imageSource={iconSource}
        anchor={markerAnchor}
        onPress={onPress}
        zIndex={500}
        opacity={opacity}
      />
      <ClusterCountBadgeMarker
        ref={badgeMarkerRef}
        coordinate={displayCoordinate}
        count={count}
        onPress={onPress}
        opacity={opacity}
      />
    </>
  );
}
