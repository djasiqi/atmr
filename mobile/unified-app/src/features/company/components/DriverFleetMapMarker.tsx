import { useMemo } from "react";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import {
  driverFleetMarkerTitle,
  resolveDriverStatus,
} from "../utils/companyDriverMapStatus";
import { FleetMapRasterMarker } from "./maps/FleetMapRasterMarker";
import { buildFleetDriverMarkerImageSource } from "./maps/fleetNativeMarkerImage";
import type { FleetOperationalStatus } from "./maps/mapStatusTheme";

function toFleetOperationalStatus(
  category: ReturnType<typeof resolveDriverStatus>
): FleetOperationalStatus {
  if (category === "en_mission") return "on_mission";
  if (category === "offline") return "offline";
  return "available";
}

type Props = {
  driver: CompanyDriverLiveLocation;
};

/** Marqueur carte flotte — pins Lirie (PNG natif, SVG web). */
export function DriverFleetMapMarker({ driver }: Props) {
  const status = toFleetOperationalStatus(resolveDriverStatus(driver));
  const imageSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, false),
    [status]
  );

  if (!imageSource.uri?.trim()) {
    return null;
  }

  return (
    <FleetMapRasterMarker
      coordinate={{ latitude: driver.latitude, longitude: driver.longitude }}
      imageSource={imageSource}
      anchor={{ x: 0.5, y: 1 }}
      title={driverFleetMarkerTitle(driver)}
    />
  );
}
