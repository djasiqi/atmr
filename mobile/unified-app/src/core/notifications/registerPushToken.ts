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
import {
  clearPendingPushTokenRegistration,
  flushPendingPushTokenRegistrations,
  persistPendingPushTokenRegistration,
  registerWithRetry,
} from "./pendingPushTokenRegistration";

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
  onPermissionDenied?: () => void;
};

async function registerPushTokenWithPersistence(
  provider: "expo" | "fcm",
  input: { token: string; deviceId: string; platform: "ios" | "android" },
  registerFn: () => Promise<void>,
  telemetrySource: string
): Promise<void> {
  try {
    await registerWithRetry(registerFn);
    await clearPendingPushTokenRegistration(provider);
    emitDriverTelemetry("push.token.registered", {
      source: telemetrySource,
      provider,
    });
  } catch (error) {
    await persistPendingPushTokenRegistration({
      provider,
      token: input.token,
      deviceId: input.deviceId,
      platform: input.platform,
    });
    emitDriverTelemetry("push.token.register_failed", {
      source: telemetrySource,
      provider,
      reason: error instanceof Error ? error.message : "unknown",
      persisted: true,
    });
  }
}

/**
 * Enregistre Expo + FCM avec device_id stable et listeners de rotation.
 */
export function useRegisterPushTokenEffect(options: RegisterPushTokenOptions): void {
  const { enabled, fcmEnabled, callbacks, telemetrySource, onPermissionDenied } = options;
  const Notifications = getExpoNotificationsModule();

  useEffect(() => {
    if (!enabled || !Notifications || Platform.OS === "web") return;

    let cancelled = false;
    let expoSubscription: EventSubscription | null = null;

    const registerExpo = async () => {
      try {
        await flushPendingPushTokenRegistrations(callbacks);

        const perm = await Notifications.requestPermissionsAsync();
        if (!perm.granted) {
          emitDriverTelemetry("push.token.permission_denied", {
            source: telemetrySource,
          });
          onPermissionDenied?.();
          return;
        }
        const deviceId = await getStableDeviceId();
        const tokenResult = await Notifications.getExpoPushTokenAsync();
        if (!tokenResult?.data || cancelled) return;
        const platform = Platform.OS === "ios" ? "ios" : "android";
        const payload = { token: tokenResult.data, deviceId, platform };
        await registerPushTokenWithPersistence("expo", payload, () => callbacks.registerExpo(payload), telemetrySource);
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
          const platform = Platform.OS === "ios" ? "ios" : "android";
          const payload = { token: data, deviceId, platform };
          await registerPushTokenWithPersistence(
            "expo",
            payload,
            () => callbacks.registerExpo(payload),
            telemetrySource
          );
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
  }, [Notifications, callbacks, enabled, onPermissionDenied, telemetrySource]);

  useEffect(() => {
    if (!enabled || !fcmEnabled || Platform.OS === "web") return;

    let cancelled = false;
    let fcmRegistrationInFlight = false;

    const registerFcm = async (token: string) => {
      if (!token || cancelled || fcmRegistrationInFlight) return;
      fcmRegistrationInFlight = true;
      try {
        const deviceId = await getStableDeviceId();
        const platform = Platform.OS === "ios" ? "ios" : "android";
        const payload = { token, deviceId, platform };
        await registerPushTokenWithPersistence(
          "fcm",
          payload,
          () => callbacks.registerFcm(payload),
          telemetrySource
        );
      } finally {
        fcmRegistrationInFlight = false;
      }
    };

    void (async () => {
      await flushPendingPushTokenRegistrations(callbacks);
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
