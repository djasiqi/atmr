import { Platform } from "react-native";

import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";

type RasterMarkerNativeProps = { icon: number } | { icon: { uri: string; width: number; height: number } } | { image: { uri: string; width: number; height: number } };

/**
 * Résout les props natives `icon` / `image` pour un marqueur raster.
 * iOS New Arch : préfère `icon` + module Metro (`require`) pour éviter
 * l'interop legacy sur `image` + URI (Sentry finalizeUpdates / nil subview).
 */
export function resolveFleetRasterMarkerNativeProps(
  imageSource: FleetNativeMarkerImageSource
): RasterMarkerNativeProps | null {
  const uri = imageSource.uri?.trim() ?? "";
  if (!uri) return null;

  const { width, height, assetModule } = imageSource;
  const raster = { uri, width, height };
  const isDataUri = uri.startsWith("data:");

  if (assetModule != null && !isDataUri) {
    return { icon: assetModule };
  }

  return Platform.OS === "android" ? { icon: raster } : { image: raster };
}
