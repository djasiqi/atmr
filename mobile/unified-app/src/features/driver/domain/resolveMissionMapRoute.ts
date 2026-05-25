import { resolveMissionRouteLeg } from "./missionRouteMetrics";
import type { MapLatLng } from "./missionMapCoordUtils";
import type { DriverMissionStatus } from "../types";

export type MissionMapRouteMode =
  | "planned_full"
  | "live_to_pickup"
  | "live_to_dropoff"
  | "single_point";

export type MissionMapRoutePlan = {
  mode: MissionMapRouteMode;
  origin: MapLatLng | null;
  destination: MapLatLng | null;
  /** Libellé court pour le badge carte. */
  badgePrefix: string;
};

export function resolveMissionMapRoute(options: {
  status?: string | null;
  driverCoord: MapLatLng | null;
  pickupCoord: MapLatLng | null;
  dropoffCoord: MapLatLng | null;
}): MissionMapRoutePlan {
  const leg = resolveMissionRouteLeg(options.status);
  const { driverCoord, pickupCoord, dropoffCoord } = options;

  if (leg.mode === "live" && driverCoord) {
    if (leg.destination === "pickup" && pickupCoord) {
      return {
        mode: "live_to_pickup",
        origin: driverCoord,
        destination: pickupCoord,
        badgePrefix: "Vers prise en charge",
      };
    }
    if (leg.destination === "dropoff" && dropoffCoord) {
      return {
        mode: "live_to_dropoff",
        origin: driverCoord,
        destination: dropoffCoord,
        badgePrefix: "Vers destination",
      };
    }
  }

  if (pickupCoord && dropoffCoord) {
    return {
      mode: "planned_full",
      origin: pickupCoord,
      destination: dropoffCoord,
      badgePrefix: "Trajet planifié",
    };
  }

  const single = pickupCoord ?? dropoffCoord ?? driverCoord;
  return {
    mode: "single_point",
    origin: single,
    destination: single,
    badgePrefix: "Mission",
  };
}

export function isMissionMapLiveRouteStatus(
  status: DriverMissionStatus | string | null | undefined
): boolean {
  const leg = resolveMissionRouteLeg(status);
  return leg.mode === "live";
}
