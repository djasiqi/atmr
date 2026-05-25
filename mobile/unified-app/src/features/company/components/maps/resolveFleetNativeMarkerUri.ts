import { Platform } from "react-native";

import {
  buildClusterCountBadgePngUri,
  buildClusterMarkerPngUri,
  buildDriverMarkerPngUri,
  buildEtaBadgePngUri,
  buildMissionAnchorPngUri,
} from "./fleetMarkerPngEncode";
import type { FleetNativeMarkerImageSource } from "./fleetNativeMarkerImage";

/** Android Google Maps : PNG obligatoire (les SVG data-URI affichent le pin rouge par défaut). */
export function usesAndroidFleetMarkerPng(): boolean {
  return Platform.OS === "android";
}

export function resolveFleetMarkerUri(
  svgUri: string,
  source: FleetNativeMarkerImageSource,
  androidUri: string
): FleetNativeMarkerImageSource {
  if (!usesAndroidFleetMarkerPng()) {
    return source;
  }
  return {
    uri: androidUri,
    width: source.width,
    height: source.height,
  };
}

export function withAndroidDriverMarkerPng(
  svgSource: FleetNativeMarkerImageSource,
  options: { fill: string; selected: boolean; pulse: boolean; sizePx: number }
): FleetNativeMarkerImageSource {
  return resolveFleetMarkerUri(
    svgSource.uri,
    svgSource,
    buildDriverMarkerPngUri(options)
  );
}

export function withAndroidClusterMarkerPng(
  svgSource: FleetNativeMarkerImageSource,
  count: number,
  sizePx: number
): FleetNativeMarkerImageSource {
  return resolveFleetMarkerUri(svgSource.uri, svgSource, buildClusterMarkerPngUri(count, sizePx));
}

export function withAndroidClusterCountBadgePng(
  svgSource: FleetNativeMarkerImageSource,
  label: string,
  widthPx: number,
  heightPx: number
): FleetNativeMarkerImageSource {
  return resolveFleetMarkerUri(
    svgSource.uri,
    svgSource,
    buildClusterCountBadgePngUri(label, widthPx, heightPx)
  );
}

export function withAndroidMissionAnchorPng(
  svgSource: FleetNativeMarkerImageSource,
  options: {
    fill: string;
    stroke: string;
    radiusPx: number;
    selected: boolean;
    halo: boolean;
  }
): FleetNativeMarkerImageSource {
  const androidUri = buildMissionAnchorPngUri(options);
  return {
    uri: androidUri,
    width: svgSource.width,
    height: svgSource.height,
  };
}

export function withAndroidEtaBadgePng(
  svgSource: FleetNativeMarkerImageSource,
  label: string
): FleetNativeMarkerImageSource {
  if (!usesAndroidFleetMarkerPng()) {
    return svgSource;
  }
  const png = buildEtaBadgePngUri(label);
  return { uri: png.uri, width: png.width, height: png.height };
}
