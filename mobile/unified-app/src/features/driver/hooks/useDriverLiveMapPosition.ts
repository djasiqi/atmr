import { useEffect, useMemo, useState } from "react";
import {
  getDriverLastKnownPosition,
  subscribeDriverTrackingBridge,
} from "../services/driverTrackingBridge";
import { missionCoordToMapPoint, normalizeMissionMapPoint, type MapLatLng } from "../domain/missionMapCoordUtils";

/** Position chauffeur pour la carte : GPS live (tracking) prioritaire sur coords API. */
export function useDriverLiveMapPosition(
  apiLat?: unknown,
  apiLng?: unknown,
  enabled = true
): MapLatLng | null {
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

  return useMemo(() => {
    const fromLive = missionCoordToMapPoint(livePosition);
    const fromApi = normalizeMissionMapPoint(apiLat, apiLng);
    return fromLive ?? fromApi;
  }, [livePosition, apiLat, apiLng]);
}
