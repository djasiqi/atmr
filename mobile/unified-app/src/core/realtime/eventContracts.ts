export type CanonicalDriverEventType =
  | "mission_assigned"
  | "mission_updated"
  | "mission_cancelled"
  | "mission_reassigned"
  | "driver_location_batch_ack"
  | "eta_changed";

export type CanonicalCompanyEventType =
  | "booking_created"
  | "booking_updated"
  | "booking_cancelled"
  | "driver_location_update"
  | "optimizer_status_changed"
  | "delay_invalidated"
  | "booking_message_sent"
  | "team_chat_message"
  | "urgent_alert"
  | "company_dispatch_update"
  // Phase 2 PR B/C — gate D3.1
  | "dispatch_assignment"
  | "dispatch_run_lifecycle";

const DRIVER_EVENT_ALIASES: Record<string, CanonicalDriverEventType> = {
  mission_assigned: "mission_assigned",
  new_booking: "mission_assigned",
  booking_assigned: "mission_assigned",
  mission_updated: "mission_updated",
  mission_status_changed: "mission_updated",
  booking_updated: "mission_updated",
  mission_cancelled: "mission_cancelled",
  booking_cancelled: "mission_cancelled",
  mission_reassigned: "mission_reassigned",
  booking_reassigned: "mission_reassigned",
  driver_location_batch_ack: "driver_location_batch_ack",
  eta_changed: "eta_changed",
};

const COMPANY_EVENT_ALIASES: Record<string, CanonicalCompanyEventType> = {
  booking_created: "booking_created",
  new_booking: "booking_created",
  booking_updated: "booking_updated",
  mission_updated: "booking_updated",
  ride_updated: "booking_updated",
  booking_cancelled: "booking_cancelled",
  mission_cancelled: "booking_cancelled",
  ride_cancelled: "booking_cancelled",
  driver_location_update: "driver_location_update",
  driver_live_state_update: "driver_location_update",
  optimizer_status_changed: "optimizer_status_changed",
  delay_invalidated: "delay_invalidated",
  booking_message: "booking_message_sent",
  booking_message_sent: "booking_message_sent",
  team_chat_message: "team_chat_message",
  urgent_alert: "urgent_alert",
  company_dispatch_update: "company_dispatch_update",
  // Phase 2 PR B/C — gate D3.1 : dispatch_* aliases vers 2 canoniques distincts
  // pour permettre une invalidation cohérente sans surfetch.
  dispatch_assignment: "dispatch_assignment",
  dispatch_run_started: "dispatch_run_lifecycle",
  dispatch_run_completed: "dispatch_run_lifecycle",
  dispatch_run_failed: "dispatch_run_lifecycle",
};

export function normalizeDriverEventType(input: unknown): CanonicalDriverEventType | null {
  if (typeof input !== "string") return null;
  return DRIVER_EVENT_ALIASES[input] ?? null;
}

export function normalizeCompanyEventType(input: unknown): CanonicalCompanyEventType | null {
  if (typeof input !== "string") return null;
  return COMPANY_EVENT_ALIASES[input] ?? null;
}
