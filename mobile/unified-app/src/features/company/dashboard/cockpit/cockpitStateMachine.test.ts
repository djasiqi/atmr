import {
  createFsmSnapshot,
  inferFsmStateFromSignals,
  reduceCockpitFsm,
} from "./cockpitFiniteStateMachine";
import { resolveCockpitOrchestration } from "./cockpitOrchestrator";
import { capVisibleRouteCount } from "./semanticRouteSystem";
import { computeMapDensityPolicy } from "./mapDensityGovernor";

describe("cockpitFiniteStateMachine", () => {
  it("transitions to DRIVER_FOCUS on driver select", () => {
    const next = reduceCockpitFsm(createFsmSnapshot(), { type: "DRIVER_SELECT" });
    expect(next.state).toBe("DRIVER_FOCUS");
  });

  it("infers INCIDENT_ALERT when urgent without sheet", () => {
    expect(
      inferFsmStateFromSignals({
        searchActive: false,
        selectedDriverId: null,
        driverSheetOpen: false,
        filtersOpen: false,
        layersOpen: false,
        urgentCount: 2,
      })
    ).toBe("INCIDENT_ALERT");
  });
});

describe("cockpitOrchestrator", () => {
  const baseInput = {
    realtimeStatus: "healthy",
    driverSheetOpen: false,
    driverSheetSnap: "collapsed" as const,
    opsSheetOpen: false,
    filtersOpen: false,
    layersOpen: false,
    searchActive: false,
    navigationActive: false,
    selectedDriverId: null,
    selectedMissionId: null,
    urgentCount: 0,
    delayedCount: 0,
    activeDriverCount: 5,
    inProgressMissionCount: 2,
    policyFailureCount: 0,
    interactionBurstPerMinute: 0,
    healthScore: 100,
    stabilized: true,
    driverCount: 5,
  };

  it("limits routes in global vector mode", () => {
    const decision = resolveCockpitOrchestration(baseInput);
    expect(decision.globalVectorMode).toBe(true);
    expect(decision.maxVisibleRoutes).toBe(0);
  });

  it("allows up to 2 routes in driver focus", () => {
    const decision = resolveCockpitOrchestration({
      ...baseInput,
      selectedDriverId: 1,
      driverSheetOpen: true,
      fsmSnapshot: reduceCockpitFsm(createFsmSnapshot(), { type: "DRIVER_SELECT" }),
    });
    expect(decision.fsmState).toBe("DRIVER_FOCUS");
    expect(decision.maxVisibleRoutes).toBeLessThanOrEqual(2);
  });
});

describe("mapDensityGovernor", () => {
  it("enters aggregate mode above 100 drivers", () => {
    const policy = computeMapDensityPolicy({ driverCount: 120 });
    expect(policy.level).toBe("aggregate");
    expect(capVisibleRouteCount(policy, false)).toBe(1);
  });
});
