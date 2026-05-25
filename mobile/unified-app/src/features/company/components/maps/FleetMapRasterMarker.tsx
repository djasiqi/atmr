import { memo, useMemo } from "react";
import { Platform } from "react-native";
import { Marker } from "react-native-maps";

import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";
import { resolveMetroAssetSource } from "./resolveMetroAssetSource";

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
 * Marqueur raster PNG embarqué.
 * Android : `icon` avec `require()` natif (une URI data/SVG ou file:// Windows → pin rouge Google).
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
    const { width, height, assetModule, uri } = imageSource;

    if (assetModule != null) {
      if (Platform.OS === "android") {
        // require() natif : seule méthode fiable pour éviter le pin rouge Google.
        return { icon: assetModule };
      }
      const resolved = resolveMetroAssetSource(assetModule);
      return {
        image: {
          uri: resolved.uri,
          width,
          height,
        },
      };
    }

    const raster = { uri, width, height };
    return Platform.OS === "android" ? { icon: raster } : { image: raster };
  }, [imageSource]);

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
