import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";

/** Cercle PNG centré → {0.5,0.5} ; pin Lirie (asset natif) → {0.5,1}. */
export function resolveFleetMarkerAnchor(
  source: FleetNativeMarkerImageSource
): { x: number; y: number } {
  if (source.assetModule != null) {
    return { x: 0.5, y: 1 };
  }
  return { x: 0.5, y: 0.5 };
}
