import { Platform } from "react-native";

import {
  makeFleetClusterCountBadgeDataUrl,
  makeFleetEtaBadgeMarkerDataUrl,
  makeMissionAnchorMarkerDataUrl,
} from "./fleetMarkerIcons";

import type { FleetMissionAnchorStyle } from "./fleetMapMissionVisual";

import { buildLirieDriverMarkerImageSource } from "./fleetLirieDriverMarkerAssets";
import {
  pickClusterRepresentativeStatus,
  resolveFleetClusterCountBadgeLayout,
} from "./fleetLirieClusterMarker";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import {
  FLEET_NATIVE_DRIVER_MARKER_SIZE_PX as FLEET_DRIVER_MARKER_SIZE_PX,
  LIRIE_CLUSTER_MARKER_DISPLAY_WIDTH_PX,
} from "./fleetLirieMarkerSizing";
import type { FleetOperationalStatus } from "./mapStatusTheme";

import {
  usesAndroidFleetMarkerPng,
  withAndroidClusterCountBadgePng,
  withAndroidEtaBadgePng,
  withAndroidMissionAnchorPng,
} from "./resolveFleetNativeMarkerUri";



/** Largeur d’affichage des pins chauffeur Lirie sur la carte native. */
export const FLEET_NATIVE_DRIVER_MARKER_SIZE_PX = FLEET_DRIVER_MARKER_SIZE_PX;

export const FLEET_NATIVE_CLUSTER_MARKER_SIZE_PX = LIRIE_CLUSTER_MARKER_DISPLAY_WIDTH_PX;



export type FleetNativeMarkerImageSource = {
  uri: string;
  width: number;
  height: number;
  assetModule?: number;
};

/** Sur mobile, seules les URL distantes ou PNG data-URI sont sûres (pas SVG / file:// hôte). */
function isUsableNativeMarkerUri(uri: string): boolean {
  if (uri.startsWith("data:image/svg")) return false;
  if (uri.startsWith("data:image/png") || uri.startsWith("data:image/jpeg")) return true;
  if (uri.startsWith("http://") || uri.startsWith("https://")) return true;
  return false;
}

function resolveCustomDriverMarkerIconUri(): string | null {
  const raw = process.env.EXPO_PUBLIC_DRIVER_MARKER_ICON_URI?.trim();
  if (!raw) return null;

  if (Platform.OS === "android" || Platform.OS === "ios") {
    return isUsableNativeMarkerUri(raw) ? raw : null;
  }

  if (raw.includes("://") || raw.startsWith("data:image/")) {
    return raw;
  }

  if (/^[A-Za-z]:[\\/]/.test(raw)) {
    const normalized = raw.replaceAll("\\", "/");
    return `file:///${normalized}`;
  }

  if (raw.startsWith("/")) {
    return `file://${raw}`;
  }

  return raw;
}

function resolveCustomDriverMarkerIconUriByStatus(
  status: FleetOperationalStatus
): string | null {
  const statusEnvNameByKey: Record<FleetOperationalStatus, string> = {
    on_mission: "EXPO_PUBLIC_DRIVER_MARKER_ICON_ON_MISSION_URI",
    available: "EXPO_PUBLIC_DRIVER_MARKER_ICON_AVAILABLE_URI",
    break: "EXPO_PUBLIC_DRIVER_MARKER_ICON_BREAK_URI",
    delayed: "EXPO_PUBLIC_DRIVER_MARKER_ICON_DELAYED_URI",
    incident: "EXPO_PUBLIC_DRIVER_MARKER_ICON_INCIDENT_URI",
    offline: "EXPO_PUBLIC_DRIVER_MARKER_ICON_OFFLINE_URI",
  };
  const raw = process.env[statusEnvNameByKey[status]]?.trim();
  if (!raw) return null;

  if (Platform.OS === "android" || Platform.OS === "ios") {
    return isUsableNativeMarkerUri(raw) ? raw : null;
  }

  if (raw.includes("://") || raw.startsWith("data:image/")) {
    return raw;
  }
  if (/^[A-Za-z]:[\\/]/.test(raw)) {
    const normalized = raw.replaceAll("\\", "/");
    return `file:///${normalized}`;
  }
  if (raw.startsWith("/")) {
    return `file://${raw}`;
  }
  return raw;
}



export function buildFleetDriverMarkerImageSource(

  status: FleetOperationalStatus,

  selected = false

): FleetNativeMarkerImageSource {

  const sizePx = Math.round(
    FLEET_NATIVE_DRIVER_MARKER_SIZE_PX * (selected ? 1.08 : 1)
  );

  const customUri = resolveCustomDriverMarkerIconUri();
  const customStatusUri = resolveCustomDriverMarkerIconUriByStatus(status);
  const resolvedCustomUri = customStatusUri ?? customUri;
  if (resolvedCustomUri) {
    return {
      uri: resolvedCustomUri,
      width: sizePx,
      height: sizePx,
    };
  }

  return buildLirieDriverMarkerImageSource(status, sizePx);
}



/** Pastille « 2 », « 3 »… (Android : PNG). */
export function buildFleetClusterCountBadgeImageSource(
  count: number
): FleetNativeMarkerImageSource {
  const layout = resolveFleetClusterCountBadgeLayout(count);
  const source: FleetNativeMarkerImageSource = {
    uri: makeFleetClusterCountBadgeDataUrl(layout.label, layout.width, layout.height),
    width: layout.width,
    height: layout.height,
  };
  if (!usesAndroidFleetMarkerPng()) return source;
  return withAndroidClusterCountBadgePng(
    source,
    layout.label,
    layout.width,
    layout.height
  );
}

/** @deprecated Utiliser buildFleetDriverMarkerImageSource + buildFleetClusterCountBadgeImageSource */
export function buildFleetClusterMarkerImageSource(
  count: number,
  drivers: FleetDriverMapItem[] = []
): FleetNativeMarkerImageSource {
  const status = pickClusterRepresentativeStatus(drivers);
  return buildFleetDriverMarkerImageSource(status, false);
}



export function buildFleetEtaBadgeImageSource(label: string): FleetNativeMarkerImageSource {

  const badge = makeFleetEtaBadgeMarkerDataUrl(label);

  const source: FleetNativeMarkerImageSource = {

    uri: badge.uri,

    width: badge.width,

    height: badge.height,

  };

  if (!usesAndroidFleetMarkerPng()) return source;

  return withAndroidEtaBadgePng(source, label);

}



export function buildMissionAnchorImageSource(

  anchor: FleetMissionAnchorStyle,

  selected = false

): FleetNativeMarkerImageSource {

  const sizePx = anchor.radius * 2 + (selected ? 8 : 6);

  const source: FleetNativeMarkerImageSource = {

    uri: makeMissionAnchorMarkerDataUrl(anchor.fill, {

      stroke: anchor.stroke,

      radiusPx: anchor.radius,

      selected,

      halo: anchor.role === "urgent" || anchor.role === "active",

    }),

    width: sizePx,

    height: sizePx,

  };

  if (!usesAndroidFleetMarkerPng()) return source;

  return withAndroidMissionAnchorPng(source, {

    fill: anchor.fill,

    stroke: anchor.stroke,

    radiusPx: anchor.radius,

    selected,

    halo: anchor.role === "urgent" || anchor.role === "active",

  });

}


