import { Platform } from "react-native";
import { useEffect, useMemo } from "react";

import { isFeatureEnabled } from "../featureFlags/registry";
import { useSession } from "../sessionProvider";
import { useRegisterPushTokenEffect } from "../notifications/registerPushToken";
import { registerDriverPushToken } from "../../features/driver/api/driverHttp";
import { driverFcmPlatform } from "../../features/driver/firebaseMessaging";
import { setPushPermissionDenied } from "../notifications/pushPermissionState";
import { startDeviceHealthHeartbeat } from "../../features/driver/services/deviceHealthHeartbeat";

/** Enregistrement push chauffeur (Expo + FCM) — monté uniquement en contexte driver. */
export function DriverNotificationsBridge() {
  const { status, activeContext, bootstrap } = useSession();

  const contextDriverId = useMemo(() => {
    if (activeContext?.context_type !== "driver") return null;
    const raw = activeContext.context_id;
    if (typeof raw !== "string") return null;
    if (!raw.startsWith("driver:")) return null;
    const parsed = Number(raw.slice("driver:".length));
    return Number.isFinite(parsed) ? parsed : null;
  }, [activeContext?.context_id, activeContext?.context_type]);

  const driverId = useMemo(() => {
    if (contextDriverId != null) return contextDriverId;
    if (activeContext?.context_type !== "driver") return null;
    const id = Number(bootstrap?.user?.id);
    return Number.isFinite(id) ? id : null;
  }, [activeContext?.context_type, bootstrap?.user?.id, contextDriverId]);

  const enabled =
    isFeatureEnabled("driver_push_enabled") &&
    status === "ready" &&
    activeContext?.context_type === "driver" &&
    driverId != null &&
    Platform.OS !== "web";

  const callbacks = useMemo(
    () => ({
      registerExpo: async (input: {
        token: string;
        deviceId: string;
        platform: "ios" | "android";
      }) => {
        if (driverId == null) return;
        await registerDriverPushToken({
          token: input.token,
          driverId,
          deviceId: input.deviceId,
          platform: input.platform,
          provider: "expo",
        });
      },
      registerFcm: async (input: {
        token: string;
        deviceId: string;
        platform: "ios" | "android";
      }) => {
        if (driverId == null) return;
        await registerDriverPushToken({
          token: input.token,
          driverId,
          deviceId: input.deviceId,
          platform: driverFcmPlatform(),
          provider: "fcm",
        });
      },
    }),
    [driverId]
  );

  useRegisterPushTokenEffect({
    enabled,
    fcmEnabled: isFeatureEnabled("driver_fcm_native_enabled"),
    callbacks,
    telemetrySource: "driver.notifications.bridge",
    onPermissionDenied: () => setPushPermissionDenied(true),
  });

  /**
   * Heartbeat de santé tracking : signale toutes les 60 s au backend que l'app
   * driver est vivante, même quand le FGS est silencieusement coupé (Samsung).
   * Démarre dès que la session driver est prête, s'arrête au logout / changement
   * de contexte. Web exclu (gardé côté service).
   */
  useEffect(() => {
    if (!enabled) return undefined;
    if (Platform.OS === "web") return undefined;
    const stop = startDeviceHealthHeartbeat();
    return () => {
      try {
        stop();
      } catch {
        /* noop */
      }
    };
  }, [enabled]);

  return null;
}
