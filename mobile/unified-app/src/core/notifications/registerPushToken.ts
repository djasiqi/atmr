import { Platform } from "react-native";
import type { EventSubscription } from "expo-modules-core";

import {
  getDriverFcmToken,
  subscribeDriverFcmTokenRefresh,
} from "../../features/driver/firebaseMessaging";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { useEffect } from "../reactCompat";
import { getExpoNotificationsModule } from "./expoNotificationsCompat";
import { getStableDeviceId } from "./getStableDeviceId";

export type PushRegisterCallbacks = {
  registerExpo: (input: {
    token: string;
    deviceId: string;
    platform: "ios" | "android";
  }) => Promise<void>;
  registerFcm: (input: {
    token: string;
    deviceId: string;
    platform: "ios" | "android";
  }) => Promise<void>;
};

export type RegisterPushTokenOptions = {
  enabled: boolean;
  fcmEnabled: boolean;
  callbacks: PushRegisterCallbacks;
  telemetrySource: string;
};

/**
 * Enregistre Expo + FCM avec device_id stable et listeners de rotation.
 */
export function useRegisterPushTokenEffect(options: RegisterPushTokenOptions): void {
  const { enabled, fcmEnabled, callbacks, telemetrySource } = options;
  const Notifications = getExpoNotificationsModule();

  useEffect(() => {
    if (!enabled || !Notifications || Platform.OS === "web") return;

    let cancelled = false;
    let expoSubscription: EventSubscription | null = null;

    const registerExpo = async () => {
      try {
        const perm = await Notifications.requestPermissionsAsync();
        if (!perm.granted) {
          emitDriverTelemetry("push.token.permission_denied", {
            source: telemetrySource,
          });
          return;
        }
        const deviceId = await getStableDeviceId();
        const tokenResult = await Notifications.getExpoPushTokenAsync();
        if (!tokenResult?.data || cancelled) return;
        await callbacks.registerExpo({
          token: tokenResult.data,
          deviceId,
          platform: Platform.OS === "ios" ? "ios" : "android",
        });
        emitDriverTelemetry("push.token.registered", {
          source: telemetrySource,
          provider: "expo",
        });
      } catch (error) {
        emitDriverTelemetry("push.token.register_failed", {
          source: telemetrySource,
          provider: "expo",
          reason: error instanceof Error ? error.message : "unknown",
        });
      }
    };

    void registerExpo();

    expoSubscription = Notifications.addPushTokenListener?.(({ data }) => {
      if (!data || cancelled) return;
      void (async () => {
        try {
          const deviceId = await getStableDeviceId();
          await callbacks.registerExpo({
            token: data,
            deviceId,
            platform: Platform.OS === "ios" ? "ios" : "android",
          });
        } catch (error) {
          emitDriverTelemetry("push.token.register_failed", {
            source: telemetrySource,
            provider: "expo",
            reason: error instanceof Error ? error.message : "refresh",
          });
        }
      })();
    }) ?? null;

    return () => {
      cancelled = true;
      expoSubscription?.remove();
    };
  }, [Notifications, callbacks, enabled, telemetrySource]);

  useEffect(() => {
    if (!enabled || !fcmEnabled || Platform.OS === "web") return;

    let cancelled = false;

    const registerFcm = async (token: string) => {
      if (!token || cancelled) return;
      try {
        const deviceId = await getStableDeviceId();
        await callbacks.registerFcm({
          token,
          deviceId,
          platform: Platform.OS === "ios" ? "ios" : "android",
        });
        emitDriverTelemetry("push.token.registered", {
          source: telemetrySource,
          provider: "fcm",
        });
      } catch (error) {
        emitDriverTelemetry("push.token.register_failed", {
          source: telemetrySource,
          provider: "fcm",
          reason: error instanceof Error ? error.message : "unknown",
        });
      }
    };

    void (async () => {
      const token = await getDriverFcmToken();
      if (token) await registerFcm(token);
    })();

    const unsubscribeRefresh = subscribeDriverFcmTokenRefresh((token) => {
      void registerFcm(token);
    });

    return () => {
      cancelled = true;
      unsubscribeRefresh();
    };
  }, [callbacks, enabled, fcmEnabled, telemetrySource]);
}
