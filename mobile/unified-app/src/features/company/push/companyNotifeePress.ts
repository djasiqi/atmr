import AsyncStorage from "@react-native-async-storage/async-storage";

import { STORAGE_KEYS } from "../../../core/storage/storageKeys";
import { canUseNotifee, loadNotifee } from "../../driver/notifeeCompat";

const PRESS_TTL_MS = 5 * 60 * 1000;

const INSTITUTION_PUSH_TYPES = new Set([
  "new_request",
  "request_updated",
  "offer_unavailable",
]);

type PendingPressRecord = {
  data: Record<string, unknown>;
  savedAt: number;
};

export function isCompanyInstitutionPushPayload(
  data: Record<string, unknown>
): boolean {
  const type = typeof data.type === "string" ? data.type : null;
  if (type && INSTITUTION_PUSH_TYPES.has(type)) return true;
  const role = typeof data.recipient_role === "string" ? data.recipient_role : null;
  return role === "company" && data.offer_id != null;
}

export async function persistPendingCompanyPushPress(
  data: Record<string, unknown>
): Promise<void> {
  const record: PendingPressRecord = { data, savedAt: Date.now() };
  await AsyncStorage.setItem(
    STORAGE_KEYS.PENDING_COMPANY_PUSH_PRESS,
    JSON.stringify(record)
  );
}

export async function consumePendingCompanyPushPress(): Promise<Record<
  string,
  unknown
> | null> {
  const raw = await AsyncStorage.getItem(STORAGE_KEYS.PENDING_COMPANY_PUSH_PRESS);
  if (!raw) return null;
  await AsyncStorage.removeItem(STORAGE_KEYS.PENDING_COMPANY_PUSH_PRESS);
  try {
    const parsed = JSON.parse(raw) as PendingPressRecord;
    if (!parsed?.data || typeof parsed.data !== "object") return null;
    if (parsed.savedAt != null && Date.now() - parsed.savedAt > PRESS_TTL_MS) {
      return null;
    }
    return parsed.data;
  } catch {
    return null;
  }
}

/** Enregistré au démarrage (index.js) — persiste le tap Notifee si l'app était tuée. */
export function registerCompanyNotifeeBackgroundPressHandler(): void {
  if (!canUseNotifee()) return;
  void (async () => {
    const mod = await loadNotifee();
    if (!mod) return;
    const { default: notifee, EventType } = mod;
    notifee.onBackgroundEvent(async ({ type, detail }) => {
      if (type !== EventType.PRESS && type !== EventType.ACTION_PRESS) return;
      const rawData = detail.notification?.data;
      if (!rawData || typeof rawData !== "object") return;
      const data = rawData as Record<string, unknown>;
      if (!isCompanyInstitutionPushPayload(data)) return;
      await persistPendingCompanyPushPress(data);
    });
  })();
}

export async function registerCompanyNotifeeForegroundPressHandler(
  onPress: (data: Record<string, unknown>) => void
): Promise<(() => void) | null> {
  const mod = await loadNotifee();
  if (!mod) return null;
  const { default: notifee, EventType } = mod;
  return notifee.onForegroundEvent(({ type, detail }) => {
    if (type !== EventType.PRESS && type !== EventType.ACTION_PRESS) return;
    const rawData = detail.notification?.data;
    if (!rawData || typeof rawData !== "object") return;
    const data = rawData as Record<string, unknown>;
    if (!isCompanyInstitutionPushPayload(data)) return;
    onPress(data);
  });
}
