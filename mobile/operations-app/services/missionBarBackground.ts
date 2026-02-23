import notifee, { EventType, type Event } from "@notifee/react-native";
import { getLogger } from "@/utils/logger";
import { MissionStateManager, type MissionBarStatus } from "./missionState";
import { showMissionNotification, dismissMissionNotification } from "./missionBarAndroid";
import { safeCall } from "./deepLinks";

const log = getLogger("MissionBg");

let registered = false;

const ACTION_TO_STATUS: Record<string, MissionBarStatus> = {
  MISSION_STATUS_EN_ROUTE: "EN_ROUTE",
  MISSION_STATUS_IN_PROGRESS: "IN_PROGRESS",
  MISSION_STATUS_COMPLETED: "COMPLETED",
  EN_ROUTE: "EN_ROUTE",
  IN_PROGRESS: "IN_PROGRESS",
  COMPLETED: "COMPLETED",
};

const actionDebounce = new Map<string, number>();
const DEBOUNCE_MS = 1500;

function isDebouncedAction(actionId: string): boolean {
  const key = `${actionId}:${MissionStateManager.getState().activeMission?.id ?? "?"}`;
  const last = actionDebounce.get(key);
  const now = Date.now();
  if (last && now - last < DEBOUNCE_MS) return true;
  actionDebounce.set(key, now);
  return false;
}

/**
 * Wrapper avec timeout pour éviter les Background ANR sur Android.
 * Android donne ~5-10s max pour un background event handler.
 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`background handler timeout (${ms}ms)`)), ms)
    ),
  ]);
}

async function handleBackgroundEventInner({ type, detail }: Event): Promise<void> {
  if (type === EventType.DISMISSED) return;

  if (type === EventType.ACTION_PRESS) {
    const actionId = detail.pressAction?.id;
    if (!actionId) return;

    if (actionId === "open-quick-actions") return;

    if (isDebouncedAction(actionId)) return;

    await MissionStateManager.ensureHydrated({ skipNetwork: true });

    if (actionId === "CALL") {
      const phone = MissionStateManager.getCallablePhone();
      if (phone) await safeCall(phone);
      return;
    }

    const targetStatus = ACTION_TO_STATUS[actionId];
    if (targetStatus) {
      const bookingId = MissionStateManager.getState().activeMission?.id ?? "?";
      log.info("action press", {
        event: "action_press",
        booking_id: bookingId,
        status: targetStatus,
        source: "headless",
      });
      const ok = await MissionStateManager.requestTransition(targetStatus);
      log.info("action result", {
        event: "action_result",
        booking_id: bookingId,
        status: targetStatus,
        result: ok ? "queued" : "failed",
      });
      if (ok) {
        await showMissionNotification(MissionStateManager.getState());
      }
      if (targetStatus === "COMPLETED") {
        await MissionStateManager.stopMission();
        await dismissMissionNotification();
      }
    }
  }
}

async function handleBackgroundEvent(event: Event): Promise<void> {
  try {
    await withTimeout(handleBackgroundEventInner(event), 4000);
  } catch (e) {
    log.warn("background event handler aborted", {
      error: e instanceof Error ? e.message : String(e),
      type: event.type,
      actionId: event.detail?.pressAction?.id,
    });
  }
}

export function registerNotifeeBackgroundHandler(): void {
  if (registered) return;
  registered = true;
  notifee.onBackgroundEvent(handleBackgroundEvent);
}

export function registerNotifeeForegroundHandler(): () => void {
  return notifee.onForegroundEvent(handleBackgroundEvent);
}
