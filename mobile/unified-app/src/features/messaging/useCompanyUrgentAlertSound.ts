import { useEffect } from "react";
import { useSession } from "../../core/sessionProvider";
import { contextRealtimeRouter } from "../../core/realtime/contextRealtimeRouter";
import { normalizeCompanyEventType } from "../../core/realtime/eventContracts";
import { getActiveScreenState } from "../../core/notifications/activeScreenStore";
import { isUrgentHubMessage, playUrgentAlertSound } from "./urgentAlertSound";

function eventTargetsActiveThread(payload: Record<string, unknown>): boolean {
  const activeThreadId = getActiveScreenState().currentThreadId;
  if (!activeThreadId) return false;
  const threadId =
    typeof payload.thread_id === "string"
      ? payload.thread_id
      : typeof payload.threadId === "string"
        ? payload.threadId
        : typeof (payload.message as Record<string, unknown> | undefined)?.thread_id === "string"
          ? String((payload.message as Record<string, unknown>).thread_id)
          : payload.booking_id != null
            ? `mission:${payload.booking_id}`
            : null;
  return Boolean(threadId && threadId === activeThreadId);
}

/** Joue une alerte sonore quand l'exploitation reçoit un signalement urgence chauffeur. */
export function useCompanyUrgentAlertSound(): void {
  const { activeContext } = useSession();
  const companyContextId =
    activeContext?.context_type === "company" ? activeContext.context_id : null;

  useEffect(() => {
    if (!companyContextId) return;
    return contextRealtimeRouter.subscribe(companyContextId, (event) => {
      if (!event || typeof event !== "object") return;
      const payload = event as Record<string, unknown>;
      const eventType =
        normalizeCompanyEventType(payload.event_type) ?? String(payload.event_type ?? "");

      if (eventType === "urgent_alert") {
        void playUrgentAlertSound();
        return;
      }
      if (eventType === "team_chat_message" && isUrgentHubMessage(payload)) {
        if (eventTargetsActiveThread(payload)) return;
        void playUrgentAlertSound();
      }
    });
  }, [companyContextId]);
}
