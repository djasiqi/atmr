import { useCallback, useEffect, useMemo, useState } from "react";
import { AppState, Platform } from "react-native";
import NetInfo from "@react-native-community/netinfo";
import * as Location from "expo-location";
import { useSession } from "../../../core/sessionProvider";
import {
  getPushPermissionDenied,
  subscribePushPermissionDenied,
} from "../../../core/notifications/pushPermissionState";
import {
  resolvePushRegistrationBannerState,
  subscribePushRegistrationRefresh,
  type PushRegistrationBannerState,
} from "../../../core/notifications/pushRegistrationState";
import { useSocketStatus } from "./useSocketStatus";
import {
  checkBatteryOptimizationStatus,
  getOemBatteryGuidance,
} from "../services/batteryOptimization";
import {
  collectDriverStatusIssues,
  resolveDriverStatusAreaView,
  type DriverStatusIssue,
  type DriverStatusAreaView,
} from "../components/driverHubStatusModel";

type Options = {
  hideTrackingPrepDuplicates?: boolean;
  trackingNeedsAttention?: boolean;
};

/** Collecte les flags async existants — aucune logique GPS / push / FGS nouvelle. */
export function useDriverStatusIssues(options: Options = {}): {
  issues: DriverStatusIssue[];
  view: DriverStatusAreaView;
  refreshBattery: () => Promise<void>;
} {
  const hideTrackingPrepDuplicates = Boolean(options.hideTrackingPrepDuplicates);
  const trackingNeedsAttention = Boolean(options.trackingNeedsAttention);
  const { status } = useSession();
  const socketStatus = useSocketStatus();
  const [isOffline, setIsOffline] = useState(false);
  const [gpsEnabled, setGpsEnabled] = useState(true);
  const [pushPermissionDenied, setPushPermissionDeniedState] = useState(getPushPermissionDenied());
  const [pushRegistrationState, setPushRegistrationState] =
    useState<PushRegistrationBannerState>("ok");
  const [batteryOptimizationActive, setBatteryOptimizationActive] = useState(false);
  const [oemGuidance] = useState(() => getOemBatteryGuidance());

  useEffect(() => {
    return subscribePushPermissionDenied(() => {
      setPushPermissionDeniedState(getPushPermissionDenied());
    });
  }, []);

  useEffect(() => {
    const refreshPushState = () => {
      void resolvePushRegistrationBannerState().then(setPushRegistrationState);
    };
    refreshPushState();
    const unsub = subscribePushRegistrationRefresh(refreshPushState);
    const appSub = AppState.addEventListener("change", (next) => {
      if (next === "active") refreshPushState();
    });
    return () => {
      unsub();
      appSub.remove();
    };
  }, []);

  const refreshBattery = useCallback(async () => {
    if (Platform.OS !== "android") return;
    const result = await checkBatteryOptimizationStatus();
    setBatteryOptimizationActive(result.checked && result.isIgnoring === false);
  }, []);

  useEffect(() => {
    if (Platform.OS !== "android") return undefined;
    void refreshBattery();
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refreshBattery();
    });
    return () => {
      sub.remove();
    };
  }, [refreshBattery]);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      const connected = Boolean(state.isConnected) && state.isInternetReachable !== false;
      setIsOffline(!connected);
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const enabled = await Location.hasServicesEnabledAsync();
        if (mounted) setGpsEnabled(enabled);
      } catch {
        if (mounted) setGpsEnabled(false);
      }
    };
    void tick();
    const interval = setInterval(() => void tick(), 10_000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const issues = useMemo(
    () =>
      collectDriverStatusIssues({
        hideTrackingPrepDuplicates,
        trackingNeedsAttention,
        pushDisclosure: pushRegistrationState === "disclosure_required",
        pushPending: pushRegistrationState === "registration_pending",
        pushFailed: pushRegistrationState === "registration_failed",
        pushDenied: pushPermissionDenied,
        offline: isOffline,
        socketDegraded: socketStatus.degraded && socketStatus.connected,
        gpsDisabled: !gpsEnabled,
        batteryOptimization: batteryOptimizationActive,
        oemRequired: oemGuidance.hasOemSettings && batteryOptimizationActive,
        oemManufacturer: oemGuidance.manufacturer,
        sessionError: status === "error",
      }),
    [
      hideTrackingPrepDuplicates,
      trackingNeedsAttention,
      pushRegistrationState,
      pushPermissionDenied,
      isOffline,
      socketStatus.degraded,
      socketStatus.connected,
      gpsEnabled,
      batteryOptimizationActive,
      oemGuidance.hasOemSettings,
      oemGuidance.manufacturer,
      status,
    ]
  );

  return {
    issues,
    view: resolveDriverStatusAreaView(issues),
    refreshBattery,
  };
}
