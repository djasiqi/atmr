import { Platform } from "react-native";
import { useMemo } from "react";

import { isFeatureEnabled } from "../featureFlags/registry";
import { useSession } from "../sessionProvider";
import { useRegisterPushTokenEffect } from "../notifications/registerPushToken";
import { registerDriverPushToken } from "../../features/driver/api/driverHttp";
import { driverFcmPlatform } from "../../features/driver/firebaseMessaging";

/** Enregistrement push chauffeur (Expo + FCM) — monté uniquement en contexte driver. */
export function DriverNotificationsBridge() {
  const { status, activeContext, bootstrap } = useSession();

  const driverId = useMemo(() => {
    if (activeContext?.context_type !== "driver") return null;
    const id = Number(bootstrap?.user?.id);
    return Number.isFinite(id) ? id : null;
  }, [activeContext?.context_type, bootstrap?.user?.id]);

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
  });

  return null;
}
