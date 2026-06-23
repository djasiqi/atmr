import { forwardRef, memo, useMemo } from "react";
import { Marker } from "react-native-maps";

import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";
import { isValidMapCoord } from "./mapsIosNewArchSafeMode";
import { resolveFleetRasterMarkerNativeProps } from "./resolveFleetRasterMarkerNativeProps";

type Props = {
  coordinate: { latitude: number; longitude: number };
  imageSource: FleetNativeMarkerImageSource;
  anchor?: { x: number; y: number };
  zIndex?: number;
  opacity?: number;
  title?: string;
  onPress?: (e: { stopPropagation?: () => void }) => void;
};

const DEFAULT_ANCHOR = { x: 0.5, y: 0.5 } as const;

/**
 * Marqueur raster — consommateur pur : utilise uniquement uri/width/height déjà résolus.
 * Ne refait pas de résolution Metro au render (uri déjà fournie par les builders).
 */
const FleetMapRasterMarkerComponent = forwardRef<Marker, Props>(function FleetMapRasterMarkerComponent(
  {
    coordinate,
    imageSource,
    anchor = DEFAULT_ANCHOR,
    zIndex,
    opacity = 1,
    title,
    onPress,
  },
  ref
) {
  const markerProps = useMemo(
    () => resolveFleetRasterMarkerNativeProps(imageSource),
    [imageSource]
  );

  if (!markerProps) {
    return null;
  }

  if (!isValidMapCoord(coordinate.latitude, coordinate.longitude)) {
    return null;
  }

  return (
    <Marker
      ref={ref}
      coordinate={coordinate}
      anchor={anchor}
      tracksViewChanges={false}
      zIndex={zIndex}
      opacity={opacity}
      title={title}
      onPress={onPress}
      {...markerProps}
    />
  );
});

export const FleetMapRasterMarker = memo(FleetMapRasterMarkerComponent);
