export type RealtimeSurface = "driver" | "company" | "client" | "institution";

export const CONTEXT_REALTIME_CHANNELS: Record<RealtimeSurface, string[]> = {
  driver: [
    "driver_mission_event",
    "team_chat_message",
    "team_chat_typing",
    "conversation_message",
  ],
  company: [
    "company_dispatch_update",
    "new_booking",
    "booking_updated",
    "booking_cancelled",
    "booking_message",
    "booking_message_sent",
    "team_chat_message",
    "conversation_message",
    "urgent_alert",
    "driver_location_update",
    "driver_live_state_update",
    "optimizer_status_changed",
    "delay_invalidated",
    // Phase 2 PR B/C — gate D3.1 (sans ces channels, ws-service confirmed_critical_miss est faussement élevé)
    "dispatch_assignment",
    "dispatch_run_started",
    "dispatch_run_completed",
    "dispatch_run_failed",
    "institution_offer_updated",
    "new_company_notification",
  ],
  client: ["client_booking_update"],
  institution: ["institution_transport_update"],
};

export function getRealtimeChannelsForSurface(surface: RealtimeSurface): string[] {
  return CONTEXT_REALTIME_CHANNELS[surface];
}
