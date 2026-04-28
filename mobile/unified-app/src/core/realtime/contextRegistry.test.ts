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
    expect(getRealtimeChannelsForSurface("driver")).toEqual(["driver_mission_event"]);
    expect(getRealtimeChannelsForSurface("company")).toEqual([
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
    ]);
  });
});
