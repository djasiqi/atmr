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

  it("labels Position confirmée only when seq+event match", () => {
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
    expect(label).toContain("Position confirmée");
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
    expect(label).toContain("Synchronisation");
    expect(label).not.toContain("Position confirmée");
  });

  it("shows Synchronisation on rejected for current attempt", () => {
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
    expect(label).toContain("Synchronisation");
    expect(label).not.toContain("Non confirmé");
  });

  it("Q1 smoking gun: ingested_non_persisted → Synchronisation…", () => {
    const fields = resolveBridgeAckFields(
      "ingested_non_persisted",
      "2026-08-17T14:00:00.000Z"
    );
    expect(fields.lastAckError).toBe("ack_ingested_non_persisted");
    expect(fields.lastAckAt).toBeNull();
    expect(fields.lastAckIsQueued).toBe(false);

    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: true,
      lastUpdate: Date.parse("2026-08-17T14:00:00.000Z"),
      lastAckAt: undefined,
      lastAckIsQueued: fields.lastAckIsQueued,
      lastAckStatus: fields.lastAckStatus,
      lastAckError: fields.lastAckError,
      currentAttemptSeq: 85,
      lastAckAttemptSeq: 85,
      currentAttemptEventId: "trk_1786978639041_1oxvf8se",
      lastAckEventId: "trk_1786978639041_1oxvf8se",
      formatSyncTime: () => "16:00",
    });
    expect(label).toBe("GPS actif · Synchronisation…");
  });

  it("Q1: queued → Synchronisation…", () => {
    const fields = resolveBridgeAckFields("queued", "2026-08-17T14:00:00.000Z");
    expect(fields.lastAckError).toBeNull();
    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: true,
      lastUpdate: Date.parse("2026-08-17T14:00:00.000Z"),
      lastAckAt: Date.parse(fields.lastAckAt!),
      lastAckIsQueued: fields.lastAckIsQueued,
      lastAckStatus: fields.lastAckStatus,
      lastAckError: fields.lastAckError,
      currentAttemptSeq: 1,
      lastAckAttemptSeq: 1,
      currentAttemptEventId: "eid-1",
      lastAckEventId: "eid-1",
      formatSyncTime: () => "16:00",
    });
    expect(label).toContain("Synchronisation");
  });

  it("BLOCKED → autorisation requise", () => {
    const label = formatBridgeSyncLabel({
      gpsEnabled: true,
      isTracking: false,
      trackingBlocked: true,
      lastUpdate: undefined,
      lastAckAt: undefined,
      lastAckIsQueued: false,
      lastAckStatus: null,
      lastAckError: null,
      currentAttemptSeq: 0,
      lastAckAttemptSeq: null,
      currentAttemptEventId: null,
      lastAckEventId: null,
      formatSyncTime: () => "",
    });
    expect(label).toContain("AUTORISATION REQUISE");
  });
});
