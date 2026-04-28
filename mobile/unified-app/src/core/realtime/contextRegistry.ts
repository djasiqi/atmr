export type RealtimeSurface = "driver" | "company" | "client" | "institution";

export const CONTEXT_REALTIME_CHANNELS: Record<RealtimeSurface, string[]> = {
  driver: ["driver_mission_event"],
  company: [
    "company_dispatch_update",
    "new_booking",
    "booking_updated",
    "booking_cancelled",
    "booking_message",
    "booking_message_sent",
    "team_chat_message",
    "driver_location_update",
    "driver_live_state_update",
    "optimizer_status_changed",
    "delay_invalidated",
  ],
  client: ["client_booking_update"],
  institution: ["institution_transport_update"],
};

export function getRealtimeChannelsForSurface(surface: RealtimeSurface): string[] {
  return CONTEXT_REALTIME_CHANNELS[surface];
}
