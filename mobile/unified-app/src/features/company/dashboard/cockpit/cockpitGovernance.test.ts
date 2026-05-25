import { describe, expect, it } from "@jest/globals";
import { computeCockpitUiState } from "./cockpitGovernance";
import type { CockpitRuntimeSignals } from "./cockpitTypes";

const baseSignals = (): CockpitRuntimeSignals => ({
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
  delayedCount: 0,
  activeDriverCount: 4,
  inProgressMissionCount: 1,
  policyFailureCount: 0,
  interactionBurstPerMinute: 0,
  healthScore: 90,
  stabilized: true,
});

describe("computeCockpitUiState", () => {
  it("démarre en mode normal avec budget visuel", () => {
    const state = computeCockpitUiState({
      ...baseSignals(),
      bootMs: Date.now() - 2000,
    });
    expect(state.mode).toBe("normal");
    expect(state.majorOverlayLayers).toBeLessThanOrEqual(2);
  });

  it("active chaos_overload si trop d’urgences", () => {
    const state = computeCockpitUiState({
      ...baseSignals(),
      urgentCount: 5,
      bootMs: Date.now() - 2000,
    });
    expect(state.mode).toBe("chaos_overload");
  });

  it("kill switch force baseline", () => {
    const state = computeCockpitUiState({
      ...baseSignals(),
      override: { killSwitch: true },
      bootMs: Date.now() - 2000,
    });
    expect(state.mode).toBe("baseline");
    expect(state.baselineOnly).toBe(true);
  });

  it("masque overlays secondaires quand sheet chauffeur ouverte", () => {
    const state = computeCockpitUiState({
      ...baseSignals(),
      driverSheetOpen: true,
      selectedDriverId: 12,
      bootMs: Date.now() - 2000,
    });
    expect(state.overlays.quickTools).toBe(false);
    expect(state.overlays.statusPills).toBe(false);
  });

  it("active simplifyClustering en safe_mode", () => {
    const state = computeCockpitUiState({
      ...baseSignals(),
      healthScore: 20,
      bootMs: Date.now() - 2000,
    });
    expect(state.mode).toBe("safe_mode");
    expect(state.simplifyClustering).toBe(true);
  });
});
