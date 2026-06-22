import { describe, expect, it } from "@jest/globals";
import {
  CONTEXT_REALTIME_CHANNELS,
  getRealtimeChannelsForSurface,
} from "./contextRegistry";

describe("context registry", () => {
  it("defines realtime channels per surface", () => {
    expect(CONTEXT_REALTIME_CHANNELS.driver).toContain("driver_mission_event");
    expect(CONTEXT_REALTIME_CHANNELS.company).toContain("company_dispatch_update");
    expect(CONTEXT_REALTIME_CHANNELS.client).toContain("client_booking_update");
    expect(CONTEXT_REALTIME_CHANNELS.institution).toContain("institution_transport_update");
  });

  it("resolves channels by surface helper", () => {
    expect(getRealtimeChannelsForSurface("driver")).toEqual([
      "driver_mission_event",
      "team_chat_message",
      "team_chat_typing",
      "conversation_message",
    ]);
    expect(getRealtimeChannelsForSurface("company")).toEqual([
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
      "dispatch_assignment",
      "dispatch_run_started",
      "dispatch_run_completed",
      "dispatch_run_failed",
      "institution_offer_updated",
      "new_company_notification",
    ]);
  });

  // Phase 2 PR B/C — gate D3.1 : sans ces channels, ws-service
  // confirmed_critical_miss est faussement élevé sur cohorte canary.
  it("subscribes company surface to dispatch_* critical channels (gate D3.1)", () => {
    const channels = getRealtimeChannelsForSurface("company");
    expect(channels).toContain("dispatch_assignment");
    expect(channels).toContain("dispatch_run_started");
    expect(channels).toContain("dispatch_run_completed");
    expect(channels).toContain("dispatch_run_failed");
  });
});
