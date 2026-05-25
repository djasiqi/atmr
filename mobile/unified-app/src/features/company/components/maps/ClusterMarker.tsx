import { useMemo } from "react";

import type { FleetDriverMapItem } from "./fleetMapTypes";
import { ClusterCountBadgeMarker } from "./ClusterCountBadgeMarker";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { buildFleetDriverMarkerImageSource } from "./fleetNativeMarkerImage";
import { pickClusterRepresentativeStatus } from "./fleetLirieClusterMarker";
import { FLEET_MAP_MARKER_DIMMED_OPACITY } from "./fleetMapTypes";

type Props = {
  latitude: number;
  longitude: number;
  count: number;
  drivers: FleetDriverMapItem[];
  onPress?: () => void;
  /** Même transparence que les chauffeurs non sélectionnés. */
  dimmed?: boolean;
};

/** Icône PNG (`require`) + chiffre en Text natif (lisible sur Android). */
export function ClusterMarker({ latitude, longitude, count, drivers, onPress, dimmed = false }: Props) {
  const opacity = dimmed ? FLEET_MAP_MARKER_DIMMED_OPACITY : 1;
  const status = useMemo(() => pickClusterRepresentativeStatus(drivers), [drivers]);
  const coordinate = useMemo(
    () => ({ latitude, longitude }),
    [latitude, longitude]
  );

  const iconSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, false),
    [status]
  );

  return (
    <>
      <FleetMapRasterMarker
        coordinate={coordinate}
        imageSource={iconSource}
        anchor={{ x: 0.5, y: 1 }}
        onPress={onPress}
        zIndex={500}
        opacity={opacity}
      />
      <ClusterCountBadgeMarker coordinate={coordinate} count={count} onPress={onPress} opacity={opacity} />
    </>
  );
}
