import { isFeatureEnabled } from "../../../core/featureFlags/registry";

export type BackgroundTrackingGate = {
  enabled: boolean;
  servicesEnabled: boolean;
  permission: "granted" | "denied" | "undetermined";
};

export async function evaluateBackgroundTrackingGate(): Promise<BackgroundTrackingGate> {
  if (!isFeatureEnabled("tracking_background_enabled")) {
    return {
      enabled: false,
      servicesEnabled: false,
      permission: "undetermined",
    };
  }
  const locationModule = await import("expo-location").catch(() => null);
  if (!locationModule) {
    return {
      enabled: true,
      servicesEnabled: false,
      permission: "undetermined",
    };
  }
  const permission = await locationModule.getBackgroundPermissionsAsync().catch(() => ({
    granted: false,
    status: "undetermined",
  }));
  const servicesEnabled = await locationModule.hasServicesEnabledAsync().catch(() => false);
  return {
    enabled: true,
    servicesEnabled,
    permission: permission.granted ? "granted" : "denied",
  };
}
