import { useMemo } from "react";
import type { CompanyDriverLiveLocation } from "../api/contracts";
import {
  driverFleetMarkerTitle,
  resolveDriverStatus,
} from "../utils/companyDriverMapStatus";
import { FleetMapRasterMarker } from "./maps/FleetMapRasterMarker";
import { buildFleetDriverMarkerImageSource } from "./maps/fleetNativeMarkerImage";
import type { FleetDriverMapItem } from "./maps/fleetMapTypes";
import type { FleetOperationalStatus } from "./maps/mapStatusTheme";

function toFleetOperationalStatus(
  category: ReturnType<typeof resolveDriverStatus>
): FleetOperationalStatus {
  if (category === "en_mission") return "busy";
  if (category === "offline") return "offline";
  if (category === "last_known") return "last_known";
  return "available";
}

function toFleetDriverMapItem(driver: CompanyDriverLiveLocation): FleetDriverMapItem {
  const operationalStatus = toFleetOperationalStatus(resolveDriverStatus(driver));
  return {
    ...driver,
    enrichment: {
      operationalStatus,
      linkedMission: null,
      delayMinutes: null,
      vehicleType: null,
      licensePlate: null,
      currentAddress: null,
      destinationAddress: null,
      etaLabel: null,
      distanceLabel: null,
      phone: null,
    },
  };
}

type Props = {
  driver: CompanyDriverLiveLocation;
};

/** Marqueur carte flotte — cercle + initiales (parité web). */
export function DriverFleetMapMarker({ driver }: Props) {
  const item = useMemo(() => toFleetDriverMapItem(driver), [driver]);
  const status = item.enrichment.operationalStatus;
  const imageSource = useMemo(
    () => buildFleetDriverMarkerImageSource(status, item),
    [item, status]
  );

  if (!imageSource.uri?.trim()) {
    return null;
  }

  return (
    <FleetMapRasterMarker
      coordinate={{ latitude: driver.latitude, longitude: driver.longitude }}
      imageSource={imageSource}
      anchor={{ x: 0.5, y: 0.5 }}
      title={driverFleetMarkerTitle(driver)}
    />
  );
}
