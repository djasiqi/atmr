import {
  applyPriorityDecayMultiplier,
  clampRouteStroke,
  FLEET_MISSION_MAP_POLICY,
  shouldShowEtaBadge,
} from "./fleetMapMissionPolicies";

describe("fleetMapMissionPolicies", () => {
  it("clamps route stroke width", () => {
    expect(clampRouteStroke(20)).toBe(FLEET_MISSION_MAP_POLICY.maxRouteStrokePx);
    expect(clampRouteStroke(0)).toBe(FLEET_MISSION_MAP_POLICY.minRouteStrokePx);
  });

  it("decays opacity progressively after unfocus", () => {
    const now = 10_000;
    const mid = applyPriorityDecayMultiplier(3, true, now, now - 260);
    const end = applyPriorityDecayMultiplier(3, true, now, now - 520);
    expect(mid).toBeLessThan(1);
    expect(end).toBe(1);
  });

  it("limits ETA badges to one on map", () => {
    expect(shouldShowEtaBadge({ isSelected: true, emphasis: 1, badgeIndex: 0 })).toBe(true);
    expect(shouldShowEtaBadge({ isSelected: false, emphasis: 4, badgeIndex: 1 })).toBe(false);
  });
});
