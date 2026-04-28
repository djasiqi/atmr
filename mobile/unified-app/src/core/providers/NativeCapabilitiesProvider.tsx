import * as Location from "expo-location";
import { isFeatureEnabled } from "../featureFlags/registry";
import { PropsWithChildren, useEffect } from "../reactCompat";
import { hydrateMissionState } from "../../features/driver/services/missionState";
import { hydrateTrackingRuntimeState } from "../../features/driver/services/trackingRuntime";
import { evaluateBackgroundTrackingGate } from "../../features/driver/services/backgroundTrackingGating";

export function NativeCapabilitiesProvider({ children }: PropsWithChildren) {
  useEffect(() => {
    void hydrateMissionState();
    void hydrateTrackingRuntimeState();
    if (!isFeatureEnabled("tracking_background_enabled")) return;
    void Location.hasServicesEnabledAsync().catch(() => undefined);
    void evaluateBackgroundTrackingGate();
  }, []);

  return children;
}
