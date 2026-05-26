import { beforeEach, describe, expect, it } from "@jest/globals";
import { missionRuntimeManager } from "./missionRuntimeManager";
import type { DriverSocketEvent } from "../types";

function recentIso(offsetMs: number): string {
  return new Date(Date.now() + offsetMs).toISOString();
}

describe("missionRuntimeManager", () => {
  beforeEach(() => {
    missionRuntimeManager.resetForTests();
  });

  it("guards transition singleflight per mission", () => {
    const first = missionRuntimeManager.beginTransition(10, "EN_ROUTE");
    const second = missionRuntimeManager.beginTransition(10, "ARRIVED");
    expect(first).toBe(true);
    expect(second).toBe(false);
    missionRuntimeManager.completeTransition(10);
    expect(missionRuntimeManager.beginTransition(10, "ARRIVED")).toBe(true);
  });

  it("deduplicates by event_id and enforces ordering", () => {
    missionRuntimeManager.registerSnapshot(7, recentIso(-120_000));

    const firstEvent: DriverSocketEvent = {
      mission_id: 7,
      event_type: "mission_updated",
      event_id: "evt-1",
      event_sequence: 1,
      updated_at: recentIso(-60_000),
    };
    expect(missionRuntimeManager.shouldApplyRealtimeEvent(firstEvent)).toEqual({
      apply: true,
      reason: "ok",
      gapDetected: false,
    });
    missionRuntimeManager.registerRealtimeEvent(firstEvent);

    expect(missionRuntimeManager.shouldApplyRealtimeEvent(firstEvent)).toEqual({
      apply: false,
      reason: "duplicate_event_id",
      gapDetected: false,
    });

    const staleBySequence: DriverSocketEvent = {
      mission_id: 7,
      event_type: "mission_updated",
      event_id: "evt-2",
      event_sequence: 1,
      updated_at: recentIso(-45_000),
    };
    expect(missionRuntimeManager.shouldApplyRealtimeEvent(staleBySequence)).toEqual({
      apply: false,
      reason: "sequence_old",
      gapDetected: false,
    });

    const gapEvent: DriverSocketEvent = {
      mission_id: 7,
      event_type: "mission_updated",
      event_id: "evt-3",
      event_sequence: 3,
      updated_at: recentIso(-15_000),
    };
    expect(missionRuntimeManager.shouldApplyRealtimeEvent(gapEvent)).toEqual({
      apply: true,
      reason: "ok",
      gapDetected: true,
    });
  });

  it("rejects stale updated_at values against known mission snapshot", () => {
    missionRuntimeManager.registerSnapshot(11, recentIso(0));
    const staleEvent: DriverSocketEvent = {
      mission_id: 11,
      event_type: "mission_updated",
      event_id: "evt-stale",
      event_sequence: 5,
      updated_at: recentIso(-1_000),
    };
    expect(missionRuntimeManager.shouldApplyRealtimeEvent(staleEvent)).toEqual({
      apply: false,
      reason: "updated_at_old",
      gapDetected: false,
    });
  });
});
