import AsyncStorage from "@react-native-async-storage/async-storage";
import { Platform } from "react-native";
import { emitDriverTelemetry } from "../observability/driverTelemetry";
import { loadNotifee } from "../../features/driver/notifeeCompat";
import { resolveDriverNotificationContract } from "../../features/driver/notificationChannels";
import { isSilentPayload, shouldSuppressVisualPush } from "../../features/driver/silentNotifications";

const STORAGE_KEY = "@atmr/push_local_display_dedup";
export const LOCAL_PUSH_DEDUP_TTL_MS = 5 * 60 * 1000;

export type LocalPushSource = "foreground" | "background" | "headless" | "fcm_callback";

type RemoteNotificationBlock = {
  title?: string;
  body?: string;
};

const memoryEntries = new Map<string, number>();
const inFlightDisplayKeys = new Set<string>();

function stableMissionDisplayDedupeKey(
  payload: Record<string, unknown>
): string | null {
  const missionId = parseMissionId(payload);
  if (missionId == null) return null;
  const type = typeof payload.type === "string" ? payload.type : null;
  if (type === "booking_assigned") {
    return `booking:${missionId}:event:assigned`;
  }
  if (type === "booking_reassigned") {
    return `booking:${missionId}:event:reassigned`;
  }
  return null;
}

function logPushEvent(
  event: string,
  fields: {
    source: LocalPushSource;
    dedupe_key: string;
    source_channel?: string;
  }
): void {
  emitDriverTelemetry(event, {
    source: "core.notifications.pushLocalDisplay",
    source_channel: fields.source_channel ?? "fcm",
    source_context: fields.source,
    dedupe_key: fields.dedupe_key,
  });
}

function parseMissionId(payload: Record<string, unknown>): number | null {
  const raw = payload.mission_id ?? payload.missionId ?? payload.booking_id ?? payload.bookingId;
  const missionId = Number(raw);
  return Number.isFinite(missionId) ? missionId : null;
}

export function buildStableDedupeKey(payload: Record<string, unknown>): string {
  const explicit =
    (typeof payload.dedupe_key === "string" && payload.dedupe_key) ||
    (typeof payload.dedupeKey === "string" && payload.dedupeKey) ||
    null;
  if (explicit) return explicit;

  const missionStableKey = stableMissionDisplayDedupeKey(payload);
  if (missionStableKey) return missionStableKey;

  const eventId =
    (typeof payload.event_id === "string" && payload.event_id) ||
    (typeof payload.eventId === "string" && payload.eventId) ||
    null;
  if (eventId) return `event:${eventId}`;

  const missionId = parseMissionId(payload);
  const type = typeof payload.type === "string" ? payload.type : null;
  if (missionId != null && type) {
    return `fallback:${type}:${missionId}`;
  }
  return `anon:${Date.now()}:${Math.random()}`;
}

async function readPersistentEntries(): Promise<Record<string, number>> {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, number>;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

async function writePersistentEntry(dedupeKey: string, expiresAt: number): Promise<void> {
  const entries = await readPersistentEntries();
  const now = Date.now();
  for (const [key, expiry] of Object.entries(entries)) {
    if (expiry <= now) {
      delete entries[key];
    }
  }
  entries[dedupeKey] = expiresAt;
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

export async function shouldSkipLocalPushDisplay(dedupeKey: string): Promise<boolean> {
  const now = Date.now();
  const memoryExpiry = memoryEntries.get(dedupeKey);
  if (memoryExpiry != null && memoryExpiry > now) {
    return true;
  }
  const persistent = await readPersistentEntries();
  const persistentExpiry = persistent[dedupeKey];
  if (persistentExpiry != null && persistentExpiry > now) {
    memoryEntries.set(dedupeKey, persistentExpiry);
    return true;
  }
  return false;
}

async function markLocalPushDisplayed(dedupeKey: string): Promise<void> {
  const expiresAt = Date.now() + LOCAL_PUSH_DEDUP_TTL_MS;
  memoryEntries.set(dedupeKey, expiresAt);
  await writePersistentEntry(dedupeKey, expiresAt);
}

function extractTitleBody(payload: Record<string, unknown>): { title: string; body: string } {
  const rawTitle =
    typeof payload.title === "string"
      ? payload.title
      : typeof (payload as { notification?: RemoteNotificationBlock }).notification?.title === "string"
        ? (payload as { notification?: RemoteNotificationBlock }).notification?.title
        : "";
  const rawBody =
    typeof payload.body === "string"
      ? payload.body
      : typeof (payload as { notification?: RemoteNotificationBlock }).notification?.body === "string"
        ? (payload as { notification?: RemoteNotificationBlock }).notification?.body
        : "";
  return {
    title: rawTitle?.trim() ?? "",
    body: rawBody?.trim() ?? "",
  };
}

export async function displayLocalDriverPush(
  payload: Record<string, unknown>,
  source: LocalPushSource,
  options?: { remoteNotification?: RemoteNotificationBlock | null }
): Promise<boolean> {
  if (Platform.OS !== "android") return false;
  if (isSilentPayload(payload)) return false;
  if (shouldSuppressVisualPush(payload)) return false;

  const dedupeKey = buildStableDedupeKey(payload);

  if (options?.remoteNotification) {
    logPushEvent("push_remote_notification_payload_detected", { source, dedupe_key: dedupeKey });
    return false;
  }

  if (inFlightDisplayKeys.has(dedupeKey)) {
    logPushEvent("push_duplicate_skipped", { source, dedupe_key: dedupeKey });
    return false;
  }
  inFlightDisplayKeys.add(dedupeKey);

  try {
    if (await shouldSkipLocalPushDisplay(dedupeKey)) {
      logPushEvent("push_duplicate_skipped", { source, dedupe_key: dedupeKey });
      return false;
    }

    const { title, body } = extractTitleBody(payload);
    if (!title && !body) {
      emitDriverTelemetry("push.notification.suppressed", {
        source: "core.notifications.pushLocalDisplay",
        suppress_reason: "empty_title_body",
        dedupe_key: dedupeKey,
        source_context: source,
      });
      return false;
    }

    const rawType = typeof payload.type === "string" ? payload.type : null;
    const contract = resolveDriverNotificationContract(rawType);
    const channelId =
      typeof payload.channelId === "string" && payload.channelId.length > 0
        ? payload.channelId
        : contract.channelId;

    const mod = await loadNotifee();
    if (!mod) return false;
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

    await markLocalPushDisplayed(dedupeKey);
    logPushEvent("push_display_local", { source, dedupe_key: dedupeKey });
    return true;
  } finally {
    inFlightDisplayKeys.delete(dedupeKey);
  }
}

export function resetPushLocalDisplayForTests(): void {
  memoryEntries.clear();
  inFlightDisplayKeys.clear();
}
