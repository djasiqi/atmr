import { getActiveScreenState } from "./activeScreenStore";
import type { DriverPushPayload } from "../../features/driver/push";
import { resolveDriverNotificationContract } from "../../features/driver/notificationChannels";
import { normalizeDriverEventType } from "../realtime/eventContracts";

export type ForegroundNotificationPresentation = {
  shouldShowBanner: boolean;
  shouldShowList: boolean;
  shouldPlaySound: boolean;
  shouldSetBadge: boolean;
  suppressReason?: string;
};

const HIDDEN_PRESENTATION: ForegroundNotificationPresentation = {
  shouldShowBanner: false,
  shouldShowList: false,
  shouldPlaySound: false,
  shouldSetBadge: false,
};

export function isCriticalNotificationType(type: string | null | undefined): boolean {
  const normalized = normalizeDriverEventType(type ?? "");
  return normalized === "mission_cancelled" || normalized === "mission_assigned";
}

export function shouldSuppressForActiveScreen(input: {
  rawType?: string | null;
  payload?: DriverPushPayload | null;
  threadId?: string | null;
}): { suppress: boolean; reason?: string } {
  const screen = getActiveScreenState();
  const rawType = (input.rawType ?? input.payload?.type ?? "").toLowerCase();

  if (input.threadId && screen.currentThreadId === input.threadId) {
    if (rawType === "chat_message" || rawType.includes("chat")) {
      return { suppress: true, reason: "active_screen" };
    }
  }

  const missionId = input.payload?.mission_id;
  if (missionId != null && screen.currentMissionId === missionId) {
    if (
      rawType === "mission_updated" ||
      rawType === "mission_reassigned" ||
      input.payload?.type === "mission_updated" ||
      input.payload?.type === "mission_reassigned"
    ) {
      return { suppress: true, reason: "active_screen" };
    }
  }

  return { suppress: false };
}

export function shouldSkipNavigationForActiveScreen(input: {
  payload: DriverPushPayload;
  threadId?: string | null;
}): boolean {
  const suppression = shouldSuppressForActiveScreen({
    rawType: input.payload.type,
    payload: input.payload,
    threadId: input.threadId,
  });
  if (suppression.suppress && input.payload.type !== "mission_cancelled") {
    return true;
  }
  return false;
}

export function resolveForegroundPresentation(input: {
  rawType?: string | null;
  payload?: DriverPushPayload | null;
  threadId?: string | null;
  silent?: boolean;
}): ForegroundNotificationPresentation {
  if (input.silent) {
    return { ...HIDDEN_PRESENTATION, suppressReason: "silent" };
  }

  const rawType = input.rawType ?? input.payload?.type ?? null;
  const contract = resolveDriverNotificationContract(rawType ?? undefined);
  if (contract.silent || contract.action === "sync_only" || rawType === "mission_refresh") {
    return { ...HIDDEN_PRESENTATION, suppressReason: "silent" };
  }

  const activeScreen = shouldSuppressForActiveScreen({
    rawType,
    payload: input.payload,
    threadId: input.threadId,
  });
  if (activeScreen.suppress) {
    return { ...HIDDEN_PRESENTATION, suppressReason: activeScreen.reason };
  }

  const critical = isCriticalNotificationType(rawType);
  return {
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: critical,
    shouldSetBadge: false,
  };
}
