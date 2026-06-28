import { useCallback, useEffect, useState } from "react";
import { AppState, Platform } from "react-native";
import * as Location from "expo-location";
import { getPushPermissionDenied } from "../../../core/notifications/pushPermissionState";
import {
  resolvePushRegistrationBannerState,
  subscribePushRegistrationRefresh,
  type PushRegistrationBannerState,
} from "../../../core/notifications/pushRegistrationState";
import { isExpoLocationPermissionGranted } from "../../../core/location/locationPermissionState";
import {
  checkBatteryOptimizationStatus,
  type BatteryOptimizationStatus,
} from "../services/batteryOptimization";
import {
  resolveDriverGpsStatus,
  resolveDriverNotificationStatus,
  resolveNotificationsEnabled,
  type DriverGpsPermissionSnapshot,
} from "./driverSettingsPresentation";

export function useDriverSettingsDeviceStatus() {
  const [gpsSnapshot, setGpsSnapshot] = useState<DriverGpsPermissionSnapshot>({
    foregroundGranted: false,
    backgroundGranted: false,
    servicesEnabled: true,
  });
  const [pushState, setPushState] = useState<PushRegistrationBannerState>("ok");
  const [pushPermissionDenied, setPushPermissionDenied] = useState(getPushPermissionDenied());
  const [batteryStatus, setBatteryStatus] = useState<BatteryOptimizationStatus>({
    isIgnoring: null,
    checked: false,
  });

  const refresh = useCallback(async () => {
    const [fg, bg, servicesEnabled, nextPush, battery] = await Promise.all([
      Location.getForegroundPermissionsAsync().catch(() => ({ granted: false })),
      Location.getBackgroundPermissionsAsync().catch(() => ({ granted: false })),
      Location.hasServicesEnabledAsync().catch(() => true),
      resolvePushRegistrationBannerState(),
      checkBatteryOptimizationStatus().catch(() => ({ isIgnoring: null, checked: false })),
    ]);
    setGpsSnapshot({
      foregroundGranted: isExpoLocationPermissionGranted(fg),
      backgroundGranted: isExpoLocationPermissionGranted(bg),
      servicesEnabled,
    });
    setPushPermissionDenied(getPushPermissionDenied());
    setPushState(nextPush);
    setBatteryStatus(battery);
  }, []);

  useEffect(() => {
    void refresh();
    const unsubPush = subscribePushRegistrationRefresh(() => {
      void refresh();
    });
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active") void refresh();
    });
    return () => {
      unsubPush();
      sub.remove();
    };
  }, [refresh]);

  const gps = resolveDriverGpsStatus(gpsSnapshot);
  const notifications = resolveDriverNotificationStatus(pushState, pushPermissionDenied);
  const notificationsEnabled = resolveNotificationsEnabled(pushState, pushPermissionDenied);
  const locationEnabled =
    gpsSnapshot.servicesEnabled && gpsSnapshot.foregroundGranted && gpsSnapshot.backgroundGranted;
  const batteryOptimizationDisabled =
    Platform.OS === "android" && batteryStatus.checked && batteryStatus.isIgnoring === true;

  return {
    refresh,
    gps,
    gpsSnapshot,
    notifications,
    notificationsEnabled,
    locationEnabled,
    batteryStatus,
    batteryOptimizationDisabled,
    pushState,
    pushPermissionDenied,
  };
}
