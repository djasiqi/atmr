import type { CompanyDriverLiveLocation } from "../../api/contracts";
import { isDriverPositionStale } from "../../utils/companyDriverMapStatus";
import type { FleetOperationalStatus } from "./mapStatusTheme";

/** Chauffeur avec position temps réel récente (connecté). */
export function isFleetDriverConnectedLive(driver: CompanyDriverLiveLocation): boolean {
  if (isDriverPositionStale(driver)) return false;
  const status = driver.location_status;
  return status === "live" || status === "recent";
}

/** Halo pulsé carte — mission active ou disponible connecté. */
export function shouldFleetMarkerLivePulse(
  operationalStatus: FleetOperationalStatus,
  driver: CompanyDriverLiveLocation
): boolean {
  if (!isFleetDriverConnectedLive(driver)) return false;
  return operationalStatus === "on_mission" || operationalStatus === "available";
}
