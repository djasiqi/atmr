/**
 * Point central : réassignation → purge mission locale, cache liste, rappels.
 * Appelé depuis socket `booking_reassigned` (et tout autre chemin équivalent).
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import { getLogger } from "@/utils/logger";
import { MissionStateManager } from "./missionState";
import { cancelMissionReminder } from "./localNotifications";

const log = getLogger("MissionReassign");

const MISSIONS_CACHE_KEY = "missions_cache_v2";

export async function handleBookingReassignedEvent(
  payload: unknown
): Promise<void> {
  const raw = payload as { booking_id?: number; id?: number };
  const bookingId = raw?.booking_id ?? raw?.id;
  if (bookingId == null || typeof bookingId !== "number") {
    log.warn("booking_reassigned missing booking_id", { payload });
    return;
  }

  log.info("handleBookingReassignedEvent", { bookingId });

  try {
    await cancelMissionReminder(bookingId);
  } catch {
    // best-effort
  }

  await MissionStateManager.onBookingReassigned(bookingId);

  try {
    const rawCache = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
    if (!rawCache) return;
    const missions = JSON.parse(rawCache) as { id: number }[];
    const filtered = missions.filter((m) => m.id !== bookingId);
    await AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(filtered));
  } catch (e) {
    log.warn("missions cache filter failed", { error: e });
  }
}
