import { Platform } from "react-native";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../core/featureFlags/registry";

type NativeFcmPayload = Record<string, unknown>;

let disposeForegroundSubscription: (() => void) | null = null;
let disposeTokenRefreshSubscription: (() => void) | null = null;

function getMessagingModule(): typeof import("@react-native-firebase/messaging").default | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    return require("@react-native-firebase/messaging").default;
  } catch {
    return null;
  }
}

export async function getDriverFcmToken(): Promise<string | null> {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return null;
  const messaging = getMessagingModule();
  if (!messaging) return null;
  try {
    await messaging().requestPermission();
    const token = await messaging().getToken();
    return token || null;
  } catch (error) {
    emitDriverTelemetry("driver.push.fcm.unavailable", {
      source: "driver.firebaseMessaging",
      reason: error instanceof Error ? error.message : "fcm_token_unavailable",
    });
    return null;
  }
}

export function subscribeDriverFcmTokenRefresh(
  onToken: (token: string) => void
): () => void {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return () => undefined;
  const messaging = getMessagingModule();
  if (!messaging) return () => undefined;
  disposeTokenRefreshSubscription?.();
  disposeTokenRefreshSubscription = messaging().onTokenRefresh((token: string) => {
    onToken(token);
  });
  return () => {
    disposeTokenRefreshSubscription?.();
    disposeTokenRefreshSubscription = null;
  };
}

export async function initDriverFirebaseMessaging(
  onDataMessage: (payload: NativeFcmPayload) => Promise<void> | void
): Promise<void> {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return;
  const messaging = getMessagingModule();
  if (!messaging) return;
  try {
    await messaging().requestPermission();
    const token = await messaging().getToken();
    emitDriverTelemetry("driver.push.fcm.token", {
      source: "driver.firebaseMessaging",
      token_present: Boolean(token),
    });
    disposeForegroundSubscription = messaging().onMessage(async (message: { data?: NativeFcmPayload }) => {
      await onDataMessage(message.data ?? {});
    });
    messaging().setBackgroundMessageHandler(async (message: { data?: NativeFcmPayload }) => {
      await onDataMessage(message.data ?? {});
    });
  } catch (error) {
    emitDriverTelemetry("driver.push.fcm.unavailable", {
      source: "driver.firebaseMessaging",
      reason: error instanceof Error ? error.message : "fcm_unavailable",
    });
  }
}

export function disposeDriverFirebaseMessaging(): void {
  if (disposeForegroundSubscription) {
    disposeForegroundSubscription();
    disposeForegroundSubscription = null;
  }
  if (disposeTokenRefreshSubscription) {
    disposeTokenRefreshSubscription();
    disposeTokenRefreshSubscription = null;
  }
}

export function driverFcmPlatform(): "ios" | "android" {
  return Platform.OS === "ios" ? "ios" : "android";
}
