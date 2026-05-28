import { memo, useMemo } from "react";
import { Platform } from "react-native";
import { Marker } from "react-native-maps";

import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";
import { isValidMapCoord } from "./mapsIosNewArchSafeMode";

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
function FleetMapRasterMarkerComponent({
  coordinate,
  imageSource,
  anchor = DEFAULT_ANCHOR,
  zIndex,
  opacity = 1,
  title,
  onPress,
}: Props) {
  const markerProps = useMemo(() => {
    const uri = imageSource.uri?.trim() ?? "";
    if (!uri) return null;

    const { width, height, assetModule } = imageSource;
    const raster = { uri, width, height };
    const isDataUri = uri.startsWith("data:");

    if (Platform.OS === "android" && assetModule != null && !isDataUri) {
      return { icon: assetModule };
    }

    return Platform.OS === "android" ? { icon: raster } : { image: raster };
  }, [imageSource]);

  if (!markerProps) {
    return null;
  }

  if (!isValidMapCoord(coordinate.latitude, coordinate.longitude)) {
    return null;
  }

  return (
    <Marker
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
}

export const FleetMapRasterMarker = memo(FleetMapRasterMarkerComponent);
