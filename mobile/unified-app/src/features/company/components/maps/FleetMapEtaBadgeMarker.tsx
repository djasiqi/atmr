import { memo, useMemo } from "react";

import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { buildFleetEtaBadgeImageSource } from "./fleetNativeMarkerImage";

type Props = {
  coordinate: { latitude: number; longitude: number };
  label: string;
  zIndex?: number;
};

function FleetMapEtaBadgeMarkerComponent({ coordinate, label, zIndex = 200 }: Props) {
  const imageSource = useMemo(() => buildFleetEtaBadgeImageSource(label), [label]);

  return (
    <FleetMapRasterMarker coordinate={coordinate} imageSource={imageSource} zIndex={zIndex} />
  );
}

export const FleetMapEtaBadgeMarker = memo(FleetMapEtaBadgeMarkerComponent);
