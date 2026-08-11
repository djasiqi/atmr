import * as Sentry from "@sentry/react-native";

import { isFeatureEnabled } from "../../../../core/featureFlags/registry";
import {
  makeFleetClusterMarkerDataUrl,
  makeFleetCircleMarkerDataUrl,
  makeFleetEtaBadgeMarkerDataUrl,
  makeMissionAnchorMarkerDataUrl,
} from "./fleetMarkerIcons";
import { buildClusterMarkerPngUri, buildDriverMarkerPngUri } from "./fleetMarkerPngEncode";

import type { FleetMissionAnchorStyle } from "./fleetMapMissionVisual";

import {
  pickClusterRepresentativeStatus,
  resolveClusterMarkerSizePx,
} from "./fleetLirieClusterMarker";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import {
  FLEET_NATIVE_DRIVER_MARKER_SIZE_PX as FLEET_DRIVER_MARKER_SIZE_PX,
} from "./fleetLirieMarkerSizing";
import type { FleetOperationalStatus } from "./mapStatusTheme";
import { resolveMarkerVisual } from "./fleetMapStatusContract";
import { resolveDriverLocationPresence } from "./driverLocationPresence";
import { isFleetDriverMarkerStale } from "./fleetMapStale";
import { driverFleetMarkerInitials } from "../../utils/companyDriverMapStatus";

import {
  usesAndroidFleetMarkerPng,
  withAndroidEtaBadgePng,
  withAndroidMissionAnchorPng,
} from "./resolveFleetNativeMarkerUri";

/** Largeur d’affichage des pins chauffeur Lirie sur la carte native. */
export const FLEET_NATIVE_DRIVER_MARKER_SIZE_PX = FLEET_DRIVER_MARKER_SIZE_PX;

export type FleetNativeMarkerImageSource = {
  uri: string;
  width: number;
  height: number;
  assetModule?: number;
};

export type BuildFleetDriverMarkerOptions = {
  isStale?: boolean;
};

function buildFleetDriverAndroidPngMarkerSource(
  status: FleetOperationalStatus,
  driver: FleetDriverMapItem,
  options?: BuildFleetDriverMarkerOptions
): FleetNativeMarkerImageSource {
  const sizePx = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
  const isStale = options?.isStale ?? isFleetDriverMarkerStale(driver);
  void isStale;
  const visual = resolveMarkerVisual(status, resolveDriverLocationPresence(driver).presence);
  const initials = driverFleetMarkerInitials(driver);
  return {
    uri: buildDriverMarkerPngUri({
      fill: visual.fill,
      opacity: visual.opacity,
      label: initials,
      sizePx,
    }),
    width: sizePx,
    height: sizePx,
  };
}

function buildFleetDriverCircleMarkerSource(
  status: FleetOperationalStatus,
  driver: FleetDriverMapItem,
  options?: BuildFleetDriverMarkerOptions
): FleetNativeMarkerImageSource {
  const sizePx = FLEET_NATIVE_DRIVER_MARKER_SIZE_PX;
  const isStale = options?.isStale ?? isFleetDriverMarkerStale(driver);
  void isStale;
  const visual = resolveMarkerVisual(status, resolveDriverLocationPresence(driver).presence);
  const initials = driverFleetMarkerInitials(driver);
  return {
    uri: makeFleetCircleMarkerDataUrl(visual.fill, sizePx, visual.opacity, {
      label: initials,
    }),
    width: sizePx,
    height: sizePx,
  };
}

export function buildFleetDriverMarkerImageSource(
  status: FleetOperationalStatus,
  driver: FleetDriverMapItem,
  options?: BuildFleetDriverMarkerOptions
): FleetNativeMarkerImageSource {
  if (isFeatureEnabled("fleet_map_safe_markers")) {
    Sentry.addBreadcrumb({
      category: "fleet_map",
      message: "fleet_map.safe_marker_mode_active",
      level: "info",
    });
  }

  try {
    if (usesAndroidFleetMarkerPng()) {
      return buildFleetDriverAndroidPngMarkerSource(status, driver, options);
    }
    return buildFleetDriverCircleMarkerSource(status, driver, options);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "build_marker_failed";
    Sentry.addBreadcrumb({
      category: "fleet_map",
      message: "fleet_map.marker_fallback_used",
      level: "warning",
      data: { status, reason },
    });
    if (usesAndroidFleetMarkerPng()) {
      return buildFleetDriverAndroidPngMarkerSource(status, driver, options);
    }
    return buildFleetDriverCircleMarkerSource(status, driver, options);
  }
}

/** Cluster web : disque coloré selon statut dominant + compteur centré. */
export function buildFleetClusterMarkerImageSource(
  count: number,
  drivers: FleetDriverMapItem[] = []
): FleetNativeMarkerImageSource {
  const status = pickClusterRepresentativeStatus(drivers);
  const visual = resolveMarkerVisual(status, false);
  const sizePx = resolveClusterMarkerSizePx(count);

  if (usesAndroidFleetMarkerPng()) {
    return {
      uri: buildClusterMarkerPngUri(count, sizePx, visual.fill),
      width: sizePx,
      height: sizePx,
    };
  }

  return {
    uri: makeFleetClusterMarkerDataUrl(count, visual.fill, sizePx),
    width: sizePx,
    height: sizePx,
  };
}

/** @deprecated Utiliser buildFleetClusterMarkerImageSource */
export function buildFleetClusterCountBadgeImageSource(
  count: number
): FleetNativeMarkerImageSource {
  return buildFleetClusterMarkerImageSource(count, []);
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

export { resolveFleetMarkerInitialsFromDisplayName as resolveDriverMarkerInitials } from "../../utils/companyDriverMapStatus";
