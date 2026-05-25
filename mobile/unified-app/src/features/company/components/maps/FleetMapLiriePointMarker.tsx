import { memo, useMemo } from "react";

import type { FleetMissionAnchorStyle } from "./fleetMapMissionVisual";
import { FleetMapRasterMarker } from "./FleetMapRasterMarker";
import { buildMissionAnchorImageSource } from "./fleetNativeMarkerImage";

type Props = {
  coordinate: { latitude: number; longitude: number };
  anchor: FleetMissionAnchorStyle;
  selected?: boolean;
};

function FleetMapLiriePointMarkerComponent({ coordinate, anchor, selected = false }: Props) {
  const imageSource = useMemo(
    () => buildMissionAnchorImageSource(anchor, selected),
    [anchor, selected]
  );

  return (
    <FleetMapRasterMarker
      coordinate={coordinate}
      imageSource={imageSource}
      zIndex={anchor.zIndex}
      opacity={anchor.opacity}
    />
  );
}

export const FleetMapLiriePointMarker = memo(FleetMapLiriePointMarkerComponent);
