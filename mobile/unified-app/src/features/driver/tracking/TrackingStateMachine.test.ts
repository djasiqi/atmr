import { describe, expect, it } from "@jest/globals";
import { resolveTrackingFsmState } from "./TrackingStateMachine";

describe("TrackingStateMachine", () => {
  it("returns MISSION_ACTIVE for live mission foreground", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: true,
        presenceEligible: false,
        appForeground: true,
        missionLive: true,
        fixStale: false,
        circuitOpen: false,
        missionTerminal: false,
      })
    ).toBe("MISSION_ACTIVE");
  });

  it("returns MISSION_RECOVERING on fix_stale during mission", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: true,
        presenceEligible: false,
        appForeground: true,
        missionLive: true,
        fixStale: true,
        circuitOpen: false,
        missionTerminal: false,
      })
    ).toBe("MISSION_RECOVERING");
  });

  it("returns PRESENCE when presenceEligible without mission (hors fenêtre FG)", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: false,
        presenceEligible: true,
        appForeground: true,
        missionLive: false,
        fixStale: false,
        circuitOpen: false,
        missionTerminal: false,
      })
    ).toBe("PRESENCE");
  });

  it("missionTerminal + enService → PRESENCE (pas IDLE)", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: false,
        presenceEligible: true,
        enService: true,
        appForeground: true,
        missionLive: false,
        fixStale: false,
        circuitOpen: false,
        missionTerminal: true,
      })
    ).toBe("PRESENCE");
  });

  it("missionTerminal + !enService → IDLE", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: false,
        presenceEligible: false,
        enService: false,
        appForeground: true,
        missionLive: false,
        fixStale: false,
        circuitOpen: false,
        missionTerminal: true,
      })
    ).toBe("IDLE");
  });

  it("blocked + enService → BLOCKED", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: false,
        presenceEligible: false,
        blocked: true,
        enService: true,
        appForeground: true,
        missionLive: false,
        fixStale: false,
        circuitOpen: false,
        missionTerminal: false,
      })
    ).toBe("BLOCKED");
  });
});
