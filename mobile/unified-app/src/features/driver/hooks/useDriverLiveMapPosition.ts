import { useEffect, useMemo, useState } from "react";
import {
  getDriverLastKnownPosition,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import {
  resolveDriverMapDisplayPosition,
  type DriverMapDisplayPosition,
} from "../domain/driverMapDisplayPosition";
import type { MapLatLng } from "../domain/missionMapCoordUtils";

/** Affichage carte : GNSS watch si présent, sinon snapshot API. Ne change pas la fraîcheur tracking. */
export function useDriverMapDisplayPosition(
  apiLat?: unknown,
  apiLng?: unknown,
  enabled = true
): DriverMapDisplayPosition {
  const [livePosition, setLivePosition] = useState(() =>
    enabled ? getDriverLastKnownPosition() : null
  );

  useEffect(() => {
    if (!enabled) {
      setLivePosition(null);
      return;
    }
    setLivePosition(getDriverLastKnownPosition());
    return subscribeDriverTrackingBridge((snapshot) => {
      setLivePosition(snapshot.lastPosition);
    });
  }, [enabled]);

  return useMemo(
    () => resolveDriverMapDisplayPosition(livePosition, apiLat, apiLng),
    [livePosition, apiLat, apiLng]
  );
}

/** Position chauffeur pour la carte (affichage uniquement). */
export function useDriverLiveMapPosition(
  apiLat?: unknown,
  apiLng?: unknown,
  enabled = true
): MapLatLng | null {
  return useDriverMapDisplayPosition(apiLat, apiLng, enabled).coord;
}
