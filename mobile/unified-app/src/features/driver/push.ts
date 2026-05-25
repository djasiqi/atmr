import {
  quickAcceptDriverMission,
  quickCompleteDriverMission,
  quickRejectDriverMission,
  quickStartDriverMission,
} from "./api";

export type DriverPushType =
  | "mission_assigned"
  | "mission_updated"
  | "mission_cancelled"
  | "mission_reassigned"
  | "reminder_action"
  | "informative";

export type DriverPushQuickAction = "accept" | "reject" | "start" | "complete";

export type DriverPushPayload = {
  type: DriverPushType;
  mission_id: number;
  event_id?: string;
  action?: DriverPushQuickAction;
  deep_link?: string;
  payload_schema?: "booking_v1" | "mission_v2" | "unknown";
  thread_id?: string;
};

export const DRIVER_PUSH_PAYLOAD_MATRIX: Record<DriverPushType, string> = {
  mission_assigned: "route_to_mission",
  mission_updated: "route_to_mission_and_reconcile",
  mission_cancelled: "route_to_mission_with_cancelled_state",
  mission_reassigned: "route_to_mission_and_force_resync",
  reminder_action: "route_to_mission_and_focus_action",
  informative: "telemetry_only_without_forced_navigation",
};

const SEEN_ACTION_TTL_MS = 60_000;
const seenPushActions = new Map<string, ReturnType<typeof setTimeout>>();

function buildActionKey(payload: DriverPushPayload) {
  return `${payload.mission_id}:${payload.event_id ?? "no-event"}:${payload.action ?? "open"}`;
}

function markSeen(actionKey: string): boolean {
  if (seenPushActions.has(actionKey)) {
    return true;
  }
  const timer = setTimeout(() => {
    seenPushActions.delete(actionKey);
  }, SEEN_ACTION_TTL_MS);
  seenPushActions.set(actionKey, timer);
  return false;
}

function clearSeen(actionKey: string): void {
  const timer = seenPushActions.get(actionKey);
  if (timer) {
    clearTimeout(timer);
    seenPushActions.delete(actionKey);
  }
}

export async function handleDriverPushQuickAction(payload: DriverPushPayload): Promise<void> {
  if (!payload.action) return;
  const actionKey = buildActionKey(payload);
  if (markSeen(actionKey)) {
    return;
  }

  try {
    if (payload.action === "accept") {
      await quickAcceptDriverMission(payload.mission_id);
      return;
    }
    if (payload.action === "reject") {
      await quickRejectDriverMission(payload.mission_id);
      return;
    }
    if (payload.action === "start") {
      await quickStartDriverMission(payload.mission_id);
      return;
    }
    if (payload.action === "complete") {
      await quickCompleteDriverMission(payload.mission_id);
      return;
    }
  } catch (error) {
    clearSeen(actionKey);
    throw error;
  }
}

export function resetSeenPushActionsForTests(): void {
  for (const timer of seenPushActions.values()) {
    clearTimeout(timer);
  }
  seenPushActions.clear();
}
