import { Platform } from "react-native";

import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import { buildDriverMarkerPngUri } from "./fleetMarkerPngEncode";
import { FLEET_STATUS_THEME } from "./mapStatusTheme";
import {
  DEFAULT_LIRIE_DRIVER_MARKER_MODULE,
  resolveLirieDriverMarkerModule,
} from "./fleetLirieDriverMarkerModules";
import {
  resolveMetroAssetSource,
  resolveMetroAssetSourceAsync,
} from "./resolveMetroAssetSource";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";

/** Ratio hauteur/largeur des PNG `driver_lirie_*` (fallback si Metro ne fournit pas la taille). */
export const LIRIE_DRIVER_PIN_ASPECT = 28 / 18;

export function usesLirieDriverMarkerRasterPng(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

export { DEFAULT_LIRIE_DRIVER_MARKER_MODULE } from "./fleetLirieDriverMarkerModules";

function buildGeneratedDriverMarkerSource(
  status: FleetOperationalStatus,
  sizePx: number,
  selected: boolean
): FleetNativeMarkerImageSource {
  const theme = FLEET_STATUS_THEME[status];
  const uri = buildDriverMarkerPngUri({
    fill: theme.fill,
    selected,
    pulse: theme.pulse === true,
    sizePx,
  });
  return {
    uri,
    width: sizePx,
    height: sizePx,
  };
}

function liriePngFromModule(
  moduleId: number,
  sizePx: number
): (FleetNativeMarkerImageSource & { assetModule?: number }) | null {
  const resolved = resolveMetroAssetSource(moduleId);
  if (!resolved?.uri) return null;

  const assetW = resolved.width ?? sizePx;
  const assetH = resolved.height ?? Math.round(sizePx * LIRIE_DRIVER_PIN_ASPECT);
  const scale = sizePx / assetW;
  return {
    uri: resolved.uri,
    width: sizePx,
    height: Math.max(1, Math.round(assetH * scale)),
    assetModule: moduleId,
  };
}

/**
 * Construit une source marqueur chauffeur Lirie.
 * Contrat : retourne toujours `{ uri, width, height }` avec uri non vide.
 * Cascade : resolvedPng ?? defaultPng ?? generatedDataUri
 */
export function buildLirieDriverMarkerImageSource(
  status: FleetOperationalStatus,
  sizePx: number,
  selected = false
): FleetNativeMarkerImageSource {
  if (isFeatureEnabled("fleet_map_safe_markers")) {
    return buildGeneratedDriverMarkerSource(status, sizePx, selected);
  }

  const statusModule = resolveLirieDriverMarkerModule(status);
  const resolvedPng = liriePngFromModule(statusModule, sizePx);
  if (resolvedPng) return resolvedPng;

  const defaultPng =
    statusModule !== DEFAULT_LIRIE_DRIVER_MARKER_MODULE
      ? liriePngFromModule(DEFAULT_LIRIE_DRIVER_MARKER_MODULE, sizePx)
      : null;
  if (defaultPng) return defaultPng;

  return buildGeneratedDriverMarkerSource(status, sizePx, selected);
}

/** Précharge les modules PNG Lirie (hors chemin render critique). */
export async function prefetchLirieDriverMarkerAssets(moduleIds: number[]): Promise<void> {
  await Promise.all(
    moduleIds.map((id) => resolveMetroAssetSourceAsync(id).catch(() => null))
  );
}
