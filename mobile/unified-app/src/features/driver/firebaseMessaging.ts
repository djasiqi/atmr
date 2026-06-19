import { Platform } from "react-native";
import { getApp } from "@react-native-firebase/app";
import {
  getMessaging,
  getToken,
  onMessage,
  onTokenRefresh,
  requestPermission,
  setBackgroundMessageHandler,
} from "@react-native-firebase/messaging";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../core/featureFlags/registry";
import { loadNotifee } from "./notifeeCompat";
import { resolveDriverNotificationContract } from "./notificationChannels";
import { isSilentPayload, shouldSuppressVisualPush } from "./silentNotifications";

type NativeFcmPayload = Record<string, unknown>;
type RemoteFcmMessage = {
  data?: NativeFcmPayload;
  notification?: {
    title?: string;
    body?: string;
  };
};

type BackgroundDataMessageHandler = (
  payload: NativeFcmPayload
) => Promise<void> | void;

let disposeForegroundSubscription: (() => void) | null = null;
let disposeTokenRefreshSubscription: (() => void) | null = null;
let backgroundHandlerRegistered = false;
let backgroundDataMessageCallback: BackgroundDataMessageHandler | null = null;

function getDriverMessagingInstance(): ReturnType<typeof getMessaging> | null {
  try {
    return getMessaging(getApp());
  } catch (error) {
    emitDriverTelemetry("driver.push.fcm.unavailable", {
      source: "driver.firebaseMessaging",
      reason: error instanceof Error ? error.message : "messaging_instance_unavailable",
      stage: "get_messaging_instance",
    });
    return null;
  }
}

function extractFcmErrorDetails(error: unknown): { reason: string; errorCode: string | null } {
  if (error instanceof Error) {
    const code =
      "code" in error && typeof (error as { code?: unknown }).code === "string"
        ? (error as { code: string }).code
        : null;
    return { reason: error.message, errorCode: code };
  }
  return { reason: "fcm_token_unavailable", errorCode: null };
}

function emitFcmUnavailable(stage: string, error?: unknown): void {
  const { reason, errorCode } =
    error !== undefined ? extractFcmErrorDetails(error) : { reason: stage, errorCode: null };
  if (__DEV__) {
    console.warn("[FCM]", stage, reason, errorCode ?? "");
  }
  emitDriverTelemetry("driver.push.fcm.unavailable", {
    source: "driver.firebaseMessaging",
    reason,
    error_code: errorCode,
    stage,
  });
}

async function reportBackgroundHandlerNoCallback(payload: NativeFcmPayload): Promise<void> {
  emitDriverTelemetry("push.fcm.background_handler_no_callback", {
    source: "driver.firebaseMessaging.background",
    platform: Platform.OS,
    payload_type: typeof payload.type === "string" ? payload.type : null,
  });
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const client = require("../../core/api/client") as {
      apiClient?: { post?: (url: string, body: unknown) => Promise<unknown> };
    };
    await client.apiClient?.post?.("/driver/me/telemetry/tracking", {
      event: "push_fcm_background_handler_no_callback",
      platform: Platform.OS,
    });
  } catch {
    /* noop */
  }
}

export function setDriverFcmBackgroundCallback(
  callback: BackgroundDataMessageHandler | null
): void {
  backgroundDataMessageCallback = callback;
}

export async function getDriverFcmToken(): Promise<string | null> {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) {
    emitFcmUnavailable("feature_flag_disabled");
    return null;
  }
  emitDriverTelemetry("driver.push.fcm.get_token_start", {
    source: "driver.firebaseMessaging",
    platform: Platform.OS,
  });
  const messagingInstance = getDriverMessagingInstance();
  if (!messagingInstance) return null;
  try {
    await requestPermission(messagingInstance);
    const token = await getToken(messagingInstance);
    if (!token) {
      emitFcmUnavailable("get_token_empty");
      return null;
    }
    emitDriverTelemetry("driver.push.fcm.token", {
      source: "driver.firebaseMessaging",
      token_present: true,
    });
    return token;
  } catch (error) {
    emitFcmUnavailable("get_token_failed", error);
    return null;
  }
}

export function subscribeDriverFcmTokenRefresh(
  onToken: (token: string) => void
): () => void {
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return () => undefined;
  const messagingInstance = getDriverMessagingInstance();
  if (!messagingInstance) return () => undefined;
  disposeTokenRefreshSubscription?.();
  disposeTokenRefreshSubscription = onTokenRefresh(messagingInstance, (token: string) => {
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
  const messagingInstance = getDriverMessagingInstance();
  if (!messagingInstance) return;
  try {
    await requestPermission(messagingInstance);
    const token = await getToken(messagingInstance);
    emitDriverTelemetry("driver.push.fcm.token", {
      source: "driver.firebaseMessaging",
      token_present: Boolean(token),
    });
    disposeForegroundSubscription = onMessage(messagingInstance, async (message: RemoteFcmMessage) => {
      await onDataMessage(message.data ?? {});
    });
    setDriverFcmBackgroundCallback(async (payload) => {
      await onDataMessage(payload);
    });
    registerDriverFcmBackgroundHandler();
  } catch (error) {
    emitFcmUnavailable("init_failed", error);
  }
}

async function displayBackgroundNotification(payload: NativeFcmPayload): Promise<void> {
  if (Platform.OS !== "android") return;
  if (isSilentPayload(payload)) return;
  if (shouldSuppressVisualPush(payload)) return;

  const rawType = typeof payload.type === "string" ? payload.type : null;
  const contract = resolveDriverNotificationContract(rawType);
  const rawTitle =
    typeof payload.title === "string"
      ? payload.title
      : typeof (payload as { notification?: { title?: string } }).notification?.title === "string"
        ? (payload as { notification?: { title?: string } }).notification?.title
        : "";
  const rawBody =
    typeof payload.body === "string"
      ? payload.body
      : typeof (payload as { notification?: { body?: string } }).notification?.body === "string"
        ? (payload as { notification?: { body?: string } }).notification?.body
        : "";
  const title = rawTitle?.trim() ?? "";
  const body = rawBody?.trim() ?? "";

  if (!title && !body) {
    emitDriverTelemetry("push.notification.suppressed", {
      source: "driver.firebaseMessaging.background",
      suppress_reason: "empty_title_body",
      payload_type: rawType ?? null,
    });
    return;
  }

  const channelId =
    typeof payload.channelId === "string" && payload.channelId.length > 0
      ? payload.channelId
      : contract.channelId;

  const mod = await loadNotifee();
  if (!mod) return;
  const { default: notifee, AndroidImportance } = mod;
  await notifee.createChannel({
    id: channelId,
    name: "Missions",
    importance: AndroidImportance.HIGH,
  });
  await notifee.displayNotification({
    title,
    body,
    data: payload as Record<string, string>,
    android: {
      channelId,
      pressAction: { id: "default" },
    },
  });
}

async function invokeBackgroundDataMessageCallback(payload: NativeFcmPayload): Promise<void> {
  const callback = backgroundDataMessageCallback;
  if (callback) {
    await callback(payload);
    return;
  }
  await reportBackgroundHandlerNoCallback(payload);
}

export function registerDriverFcmBackgroundHandler(
  onDataMessage?: BackgroundDataMessageHandler
): void {
  if (onDataMessage) {
    backgroundDataMessageCallback = onDataMessage;
  }
  if (backgroundHandlerRegistered) return;
  if (!isFeatureEnabled("driver_fcm_native_enabled")) return;
  const messagingInstance = getDriverMessagingInstance();
  if (!messagingInstance) return;
  backgroundHandlerRegistered = true;
  setBackgroundMessageHandler(messagingInstance, async (message: RemoteFcmMessage) => {
    const payload = message.data ?? {};

    const payloadType = typeof payload.type === "string" ? payload.type : null;
    if (payloadType === "silent_update" || isSilentPayload(payload)) {
      await invokeBackgroundDataMessageCallback(payload);
      emitDriverTelemetry("push.notification.suppressed", {
        source: "driver.firebaseMessaging.background",
        suppress_reason: "silent_update",
        payload_type: payloadType,
      });
      return;
    }

    await invokeBackgroundDataMessageCallback(payload);
    if (message.notification?.title || message.notification?.body) {
      return;
    }
    try {
      await displayBackgroundNotification(payload);
    } catch (error) {
      emitFcmUnavailable("background_display_failed", error);
    }
  });
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

export function __resetDriverFcmBackgroundHandlerForTests(): void {
  backgroundHandlerRegistered = false;
  backgroundDataMessageCallback = null;
}
