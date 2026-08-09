import { AppState, Platform } from "react-native";
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
  ensureNotificationDisclosureSyncedWithOsPermission,
  readNotificationDisclosureAccepted,
  subscribeNotificationDisclosureAccepted,
} from "./notificationDisclosurePersistence";
import {
  clearPendingPushTokenRegistration,
  flushPendingPushTokenRegistrations,
  persistPendingPushTokenRegistration,
  registerWithRetry,
} from "./pendingPushTokenRegistration";
import {
  clearPushRegistrationFailed,
  setPushRegistrationFailed,
} from "./pushRegistrationState";
import { reportPushRegistrationTelemetry } from "./pushRegistrationTelemetry";

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
    clearPushRegistrationFailed();
    emitDriverTelemetry("push.token.registered", {
      source: telemetrySource,
      provider,
    });
    if (provider === "fcm") {
      reportPushRegistrationTelemetry("driver_push.register_success", {
        source: telemetrySource,
        provider: "fcm",
      });
    }
  } catch (error) {
    await persistPendingPushTokenRegistration({
      provider,
      token: input.token,
      deviceId: input.deviceId,
      platform: input.platform,
    });
    setPushRegistrationFailed(true);
    emitDriverTelemetry("push.token.register_failed", {
      source: telemetrySource,
      provider,
      reason: error instanceof Error ? error.message : "unknown",
      persisted: true,
    });
  }
}

async function hasAcceptedNotificationDisclosure(telemetrySource: string): Promise<boolean> {
  await ensureNotificationDisclosureSyncedWithOsPermission();
  const disclosureAccepted = await readNotificationDisclosureAccepted();
  if (!disclosureAccepted) {
    console.info("[FCM-GATE] disclosure not accepted — push registration blocked", {
      source: telemetrySource,
    });
    reportPushRegistrationTelemetry("driver_push.disclosure_blocked", {
      source: telemetrySource,
    });
    emitDriverTelemetry("push.token.disclosure_required", {
      source: telemetrySource,
    });
  }
  return disclosureAccepted;
}

export type NotificationOsPermissionResult = {
  granted: boolean;
  status?: string;
  canAskAgain?: boolean;
};

/**
 * Seul site prod autorisé à appeler Notifications.requestPermissionsAsync
 * (gate Play `check:play-compliance`). UI settings / readiness doivent passer ici.
 */
export async function requestNotificationOsPermissionsAsync(): Promise<NotificationOsPermissionResult> {
  const Notifications = getExpoNotificationsModule();
  if (!Notifications?.requestPermissionsAsync) {
    return { granted: false, status: "unavailable" };
  }
  const perm = await Notifications.requestPermissionsAsync();
  const granted = Boolean(perm?.granted || perm?.status === "granted");
  return {
    granted,
    status: typeof perm?.status === "string" ? perm.status : undefined,
    canAskAgain: typeof perm?.canAskAgain === "boolean" ? perm.canAskAgain : undefined,
  };
}

/**
 * Enregistre Expo + FCM avec device_id stable et listeners de rotation.
 */
export function useRegisterPushTokenEffect(options: RegisterPushTokenOptions): void {
  const { enabled, fcmEnabled, callbacks, telemetrySource, onPermissionDenied } = options;
  const Notifications = getExpoNotificationsModule();
  const androidNativeFcmMode = fcmEnabled && Platform.OS === "android";

  useEffect(() => {
    if (!enabled || !Notifications || Platform.OS === "web") return;
    if (androidNativeFcmMode) return;

    let cancelled = false;
    let expoSubscription: EventSubscription | null = null;

    const registerExpo = async () => {
      try {
        if (!(await hasAcceptedNotificationDisclosure(telemetrySource))) return;

        const perm = await requestNotificationOsPermissionsAsync();
        if (!perm.granted) {
          emitDriverTelemetry("push.token.permission_denied", {
            source: telemetrySource,
          });
          onPermissionDenied?.();
          return;
        }

        await flushPendingPushTokenRegistrations(callbacks);

        const deviceId = await getStableDeviceId();
        const tokenResult = await Notifications.getExpoPushTokenAsync();
        if (!tokenResult?.data || cancelled) return;
        const platform = Platform.OS === "ios" ? "ios" : "android";
        const payload = { token: tokenResult.data, deviceId, platform };
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
          reason: error instanceof Error ? error.message : "unknown",
        });
      }
    };

    void registerExpo();

    const unsubscribeDisclosure = subscribeNotificationDisclosureAccepted(() => {
      void registerExpo();
    });

    expoSubscription = Notifications.addPushTokenListener?.(({ data }) => {
      if (!data || cancelled) return;
      void (async () => {
        try {
          if (!(await hasAcceptedNotificationDisclosure(telemetrySource))) return;
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
      unsubscribeDisclosure();
      expoSubscription?.remove();
    };
  }, [Notifications, androidNativeFcmMode, callbacks, enabled, onPermissionDenied, telemetrySource]);

  useEffect(() => {
    if (!enabled || !fcmEnabled || Platform.OS === "web") return;

    let cancelled = false;
    let fcmRegistrationInFlight = false;

    const registerExpoFallback = async (): Promise<void> => {
      if (!Notifications || cancelled) return;
      if (!(await hasAcceptedNotificationDisclosure(telemetrySource))) return;
      emitDriverTelemetry("push.token.expo_fallback_unreliable", {
        source: telemetrySource,
        platform: Platform.OS,
      });
      try {
        const perm = await Notifications.getPermissionsAsync();
        if (!perm.granted && perm.status !== "granted") return;
        const tokenResult = await Notifications.getExpoPushTokenAsync();
        if (!tokenResult?.data || cancelled) return;
        const deviceId = await getStableDeviceId();
        const platform = Platform.OS === "ios" ? "ios" : "android";
        const payload = { token: tokenResult.data, deviceId, platform };
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
          reason: error instanceof Error ? error.message : "expo_fallback_failed",
        });
      }
    };

    const registerFcm = async (token: string) => {
      if (!token || cancelled || fcmRegistrationInFlight) return;
      if (!(await hasAcceptedNotificationDisclosure(telemetrySource))) return;
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

    const attemptFcmRegistration = async (stage: string): Promise<boolean> => {
      if (cancelled) return false;
      console.info("[FCM-GATE] register effect start", {
        source: telemetrySource,
        stage,
        enabled,
        fcmEnabled,
      });
      if (!(await hasAcceptedNotificationDisclosure(telemetrySource))) return false;
      const NotificationsModule = getExpoNotificationsModule();
      if (NotificationsModule) {
        const perm = await NotificationsModule.getPermissionsAsync();
        if (!perm.granted && perm.status !== "granted") {
          console.info("[FCM-GATE] OS notification permission missing", {
            source: telemetrySource,
            stage,
            status: perm.status,
          });
          reportPushRegistrationTelemetry("driver_push.permission_blocked", {
            source: telemetrySource,
            stage,
            permission_status: perm.status,
          });
          return false;
        }
      }
      await flushPendingPushTokenRegistrations(callbacks);
      console.info("[FCM-GATE] requesting token", { source: telemetrySource, stage });
      const token = await getDriverFcmToken();
      if (token) {
        console.info("[FCM-GATE] token received", {
          source: telemetrySource,
          stage,
          tokenLength: token.length,
        });
        reportPushRegistrationTelemetry("driver_push.token_acquired", {
          source: telemetrySource,
          stage,
          provider: "fcm",
          token_length: token.length,
        });
        console.info("[FCM-GATE] posting token", { source: telemetrySource, stage, provider: "fcm" });
        await registerFcm(token);
        return true;
      }
      if (androidNativeFcmMode) {
        console.info("[FCM-GATE] getToken returned empty — expo fallback", {
          source: telemetrySource,
          stage,
        });
        emitDriverTelemetry("driver.push.fcm.unavailable", {
          source: telemetrySource,
          reason: "fcm_token_missing_after_get",
          stage,
        });
        await registerExpoFallback();
      }
      return false;
    };

    void attemptFcmRegistration("session_ready");

    const unsubscribeRefresh = subscribeDriverFcmTokenRefresh((token) => {
      void registerFcm(token);
    });
    const unsubscribeDisclosure = subscribeNotificationDisclosureAccepted(() => {
      void attemptFcmRegistration("disclosure_accepted");
    });

    const appStateSubscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active" || cancelled) return;
      void attemptFcmRegistration("app_foreground");
    });

    return () => {
      cancelled = true;
      unsubscribeRefresh();
      unsubscribeDisclosure();
      appStateSubscription.remove();
    };
  }, [Notifications, androidNativeFcmMode, callbacks, enabled, fcmEnabled, telemetrySource]);

  useEffect(() => {
    if (!enabled || Platform.OS === "web") return;
    if (androidNativeFcmMode) return;

    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState !== "active") return;
      void flushPendingPushTokenRegistrations(callbacks);
    });

    return () => {
      subscription.remove();
    };
  }, [androidNativeFcmMode, callbacks, enabled]);
}
