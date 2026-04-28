import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../core/featureFlags/registry";

type NativeFcmPayload = Record<string, unknown>;

let disposeForegroundSubscription: (() => void) | null = null;

export async function initDriverFirebaseMessaging(
  onDataMessage: (payload: NativeFcmPayload) => Promise<void> | void
): Promise<void> {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const messaging = require("@react-native-firebase/messaging").default;
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
}
