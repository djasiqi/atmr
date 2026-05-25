import { Platform } from "react-native";

import { resolveLirieDriverMarkerModule } from "./fleetLirieDriverMarkerModules";
import { resolveMetroAssetSource } from "./resolveMetroAssetSource";
import type { FleetOperationalStatus } from "./mapStatusTheme";

/** Ratio hauteur/largeur des PNG `driver_lirie_*` (fallback si Metro ne fournit pas la taille). */
export const LIRIE_DRIVER_PIN_ASPECT = 28 / 18;

export function usesLirieDriverMarkerRasterPng(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

export type LirieDriverMarkerImageSource = {
  uri: string;
  width: number;
  height: number;
  /** Module Metro (`require(png)`) — obligatoire pour Google Maps Android (sinon pin rouge). */
  assetModule: number;
};

export function buildLirieDriverMarkerImageSource(
  status: FleetOperationalStatus,
  sizePx: number
): LirieDriverMarkerImageSource {
  const assetModule = resolveLirieDriverMarkerModule(status);
  const resolved = resolveMetroAssetSource(assetModule);
  const uri = resolved?.uri?.trim();
  if (!uri) {
    throw new Error(`[fleet-map] Marqueur Lirie introuvable pour le statut « ${status} »`);
  }

  const assetW = resolved.width ?? sizePx;
  const assetH = resolved.height ?? Math.round(sizePx * LIRIE_DRIVER_PIN_ASPECT);
  const scale = sizePx / assetW;
  return {
    uri,
    width: sizePx,
    height: Math.max(1, Math.round(assetH * scale)),
    assetModule,
  };
}
