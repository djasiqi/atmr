import { describe, expect, it } from "@jest/globals";
import { resolveTrackingFsmState } from "./TrackingStateMachine";

describe("TrackingStateMachine", () => {
  it("returns MISSION_ACTIVE for live mission foreground", () => {
    expect(
      resolveTrackingFsmState({
        hasMission: true,
        presenceWindow: false,
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
        presenceWindow: false,
        appForeground: true,
        missionLive: true,
        fixStale: true,
        circuitOpen: false,
        missionTerminal: false,
      })
    ).toBe("MISSION_RECOVERING");
  });
});
