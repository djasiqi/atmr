import { describe, expect, it } from "@jest/globals";
import {
  formatBridgeSyncLabel,
  resolveBridgeAckFields,
} from "./bridgeAckSemantics";

describe("bridgeAckSemantics", () => {
  it("marks queued as Mis en file fields", () => {
    const fields = resolveBridgeAckFields("queued", "2026-07-27T12:00:00.000Z");
    expect(fields.lastAckIsQueued).toBe(true);
    expect(fields.lastAckAt).toBe("2026-07-27T12:00:00.000Z");
    expect(fields.lastAckError).toBeNull();
  });

  it("marks accepted as confirmed fields", () => {
    const fields = resolveBridgeAckFields("accepted", "2026-07-27T12:00:00.000Z");
    expect(fields.lastAckIsQueued).toBe(false);
    expect(fields.lastAckAt).toBeTruthy();
    expect(fields.lastAckError).toBeNull();
  });

  it("fail-closes on stale/rejected without confirming", () => {
    expect(resolveBridgeAckFields("stale", "t").lastAckAt).toBeNull();
    expect(resolveBridgeAckFields("rejected", "t").lastAckError).toBe("ack_rejected");
  });

  it("labels Confirmé only when seq+event match", () => {
    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: true,
      lastUpdate: Date.parse("2026-07-27T12:00:00.000Z"),
      lastAckAt: Date.parse("2026-07-27T12:00:01.000Z"),
      lastAckIsQueued: false,
      lastAckStatus: "accepted",
      lastAckError: null,
      currentAttemptSeq: 2,
      lastAckAttemptSeq: 2,
      currentAttemptEventId: "evt-b",
      lastAckEventId: "evt-b",
      formatSyncTime: () => "14:00",
    });
    expect(label).toContain("Confirmé");
  });

  it("does not confirm when event id mismatches (backlog ACK)", () => {
    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: true,
      lastUpdate: Date.parse("2026-07-27T12:00:00.000Z"),
      lastAckAt: Date.parse("2026-07-27T12:00:01.000Z"),
      lastAckIsQueued: false,
      lastAckStatus: "accepted",
      lastAckError: null,
      currentAttemptSeq: 2,
      lastAckAttemptSeq: 2,
      currentAttemptEventId: "evt-new",
      lastAckEventId: "evt-old",
      formatSyncTime: () => "14:00",
    });
    expect(label).toContain("Envoyé");
    expect(label).not.toContain("Confirmé");
  });

  it("shows Non confirmé on rejected for current attempt", () => {
    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: true,
      lastUpdate: Date.parse("2026-07-27T12:00:00.000Z"),
      lastAckAt: undefined,
      lastAckIsQueued: false,
      lastAckStatus: "rejected",
      lastAckError: "ack_rejected",
      currentAttemptSeq: 1,
      lastAckAttemptSeq: 1,
      currentAttemptEventId: "evt-1",
      lastAckEventId: "evt-1",
      formatSyncTime: () => "14:00",
    });
    expect(label).toContain("Non confirmé");
  });
});
