import { mergeFrameEvents } from "./cockpitFrameArbitration";
import { createCockpitRuntimeStore } from "./cockpitRuntimeStore";
import { resolveCockpitOrchestration } from "./cockpitOrchestrator";
import { createFsmSnapshot, reduceCockpitFsm } from "./cockpitFiniteStateMachine";
import { resolveFleetHealthScore } from "./healthScoreModel";

describe("cockpit runtime burst", () => {
  it("coalesces GPS but keeps critical events", () => {
    const now = Date.now();
    const merged = mergeFrameEvents([
      { type: "GPS_UPDATE", atMs: now, source: "gps", coalescable: true },
      { type: "GPS_UPDATE", atMs: now + 1, source: "gps", coalescable: true },
      { type: "SEARCH_OPENED", atMs: now + 2, source: "ui", coalescable: false },
      { type: "DRIVER_SELECTED", atMs: now + 3, source: "ui", coalescable: false },
    ]);
    expect(merged.filter((e) => e.type === "GPS_UPDATE")).toHaveLength(1);
    expect(merged.some((e) => e.type === "SEARCH_OPENED")).toBe(true);
    expect(merged.some((e) => e.type === "DRIVER_SELECTED")).toBe(true);
  });

  it("FSM stable after DRIVER_CLEAR", () => {
    let fsm = reduceCockpitFsm(createFsmSnapshot(), { type: "DRIVER_SELECT" });
    expect(fsm.state).toBe("DRIVER_FOCUS");
    fsm = reduceCockpitFsm(fsm, { type: "DRIVER_CLEAR" });
    expect(fsm.state).not.toBe("DRIVER_FOCUS");
  });

  it("urgentCount distinct from delayedCount in orchestration input", () => {
    const decision = resolveCockpitOrchestration({
      realtimeStatus: "healthy",
      driverSheetOpen: false,
      driverSheetSnap: "collapsed",
      opsSheetOpen: false,
      filtersOpen: false,
      layersOpen: false,
      searchActive: false,
      navigationActive: false,
      selectedDriverId: null,
      selectedMissionId: null,
      urgentCount: 0,
      delayedCount: 3,
      activeDriverCount: 20,
      inProgressMissionCount: 5,
      policyFailureCount: 0,
      interactionBurstPerMinute: 0,
      healthScore: 0,
      stabilized: true,
      driverCount: 20,
      fsmSnapshot: createFsmSnapshot(),
    });
    expect(decision.fleetHealthScore).toBeLessThan(100);
  });

  it("store flush returns enqueued frame events", () => {
    const store = createCockpitRuntimeStore();
    store.enqueueFrameEvent({
      type: "GPS_UPDATE",
      atMs: 1,
      source: "t",
      coalescable: true,
    });
    const events = store.flushFrame();
    expect(events).toHaveLength(1);
    expect(store.flushFrame()).toHaveLength(0);
  });

  it("fleet health reacts to offline", () => {
    const score = resolveFleetHealthScore({
      delayedCount: 0,
      urgentCount: 0,
      unassignedCount: 0,
      criticalEtaCount: 0,
      realtimeStatus: "offline",
      policyFailureCount: 0,
      interactionBurstPerMinute: 0,
    });
    expect(score).toBeLessThan(70);
  });
});
