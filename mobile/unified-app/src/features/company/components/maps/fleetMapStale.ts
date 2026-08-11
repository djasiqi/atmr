import type { CompanyDriverLiveLocation } from "../../api/contracts";

import { resolveDriverLocationPresence } from "./driverLocationPresence";

/**
 * Signal visuel stale / fantôme — délègue à la machine d’état présence GPS.
 */
export function isFleetDriverMarkerStale(driver: CompanyDriverLiveLocation): boolean {
  return resolveDriverLocationPresence(driver).isVisuallyStale;
}
