import type { CockpitFsmEvent, CockpitFsmSnapshot } from "./cockpitFiniteStateMachine";
import {
  createFsmSnapshot,
  inferFsmStateFromSignals,
  reduceCockpitFsm,
} from "./cockpitFiniteStateMachine";
import type { CockpitEvent } from "./cockpitEventBus";
import { loadCockpitSession } from "./cockpitSession";
import {
  createStabilityBudgetState,
  type StabilityBudgetState,
} from "./cockpitStabilityBudget";

export type CockpitRuntimeSnapshot = {
  fsm: CockpitFsmSnapshot;
  stability: StabilityBudgetState;
  sessionHydrated: boolean;
  initialSelectedDriverId: number | null;
  pendingFrameEventCount: number;
};

type BootstrapSignals = {
  searchActive: boolean;
  selectedDriverId: number | null;
  driverSheetOpen: boolean;
  filtersOpen: boolean;
  layersOpen: boolean;
  urgentCount: number;
};

/** Store léger — état runtime uniquement, aucune décision cockpit. */
export class CockpitRuntimeStore {
  private fsmSnapshot: CockpitFsmSnapshot = createFsmSnapshot();
  private stabilityState = createStabilityBudgetState();
  private frameBuffer: CockpitEvent[] = [];
  private sessionHydrated = false;
  private initialSelectedDriverId: number | null = null;

  getSnapshot(): CockpitRuntimeSnapshot {
    return {
      fsm: this.fsmSnapshot,
      stability: this.stabilityState,
      sessionHydrated: this.sessionHydrated,
      initialSelectedDriverId: this.initialSelectedDriverId,
      pendingFrameEventCount: this.frameBuffer.length,
    };
  }

  getFsmSnapshot(): CockpitFsmSnapshot {
    return this.fsmSnapshot;
  }

  getStabilityState(): StabilityBudgetState {
    return this.stabilityState;
  }

  setStabilityState(state: StabilityBudgetState): void {
    this.stabilityState = state;
  }

  dispatch(event: CockpitFsmEvent): void {
    this.fsmSnapshot = reduceCockpitFsm(this.fsmSnapshot, event);
  }

  enqueueFrameEvent(event: CockpitEvent): void {
    this.frameBuffer.push(event);
  }

  flushFrame(): CockpitEvent[] {
    const events = [...this.frameBuffer];
    this.frameBuffer = [];
    return events;
  }

  async hydrateSession(): Promise<void> {
    const snap = await loadCockpitSession();
    if (snap?.selectedDriverId != null) {
      this.initialSelectedDriverId = snap.selectedDriverId;
    }
    this.sessionHydrated = true;
  }

  bootstrapFsm(signals: BootstrapSignals): void {
    const state = inferFsmStateFromSignals(signals);
    this.fsmSnapshot = createFsmSnapshot(state);
  }

  reset(): void {
    this.fsmSnapshot = createFsmSnapshot();
    this.stabilityState = createStabilityBudgetState();
    this.frameBuffer = [];
    this.sessionHydrated = false;
    this.initialSelectedDriverId = null;
  }
}

export function createCockpitRuntimeStore(): CockpitRuntimeStore {
  return new CockpitRuntimeStore();
}
