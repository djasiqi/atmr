/** États FSM — logique de transition uniquement, sans effets UI. */
export type CockpitFsmState =
  | "GLOBAL_IDLE"
  | "GLOBAL_TRACKING"
  | "DRIVER_FOCUS"
  | "SEARCH_ACTIVE"
  | "INCIDENT_ALERT"
  | "DISPATCH_ACTION";

export type CockpitFsmEvent =
  | { type: "ENTER"; target: CockpitFsmState }
  | { type: "DRIVER_SELECT" }
  | { type: "DRIVER_CLEAR" }
  | { type: "SEARCH_OPEN" }
  | { type: "SEARCH_CLOSE" }
  | { type: "INCIDENT_RAISE" }
  | { type: "INCIDENT_ACK" }
  | { type: "DISPATCH_OPEN" }
  | { type: "DISPATCH_CLOSE" }
  | { type: "TRACKING_START" };

export type CockpitFsmSnapshot = {
  state: CockpitFsmState;
  previousState: CockpitFsmState | null;
  transitionCount: number;
};

export function createFsmSnapshot(
  state: CockpitFsmState = "GLOBAL_IDLE"
): CockpitFsmSnapshot {
  return { state, previousState: null, transitionCount: 0 };
}

function transition(
  snapshot: CockpitFsmSnapshot,
  next: CockpitFsmState
): CockpitFsmSnapshot {
  if (snapshot.state === next) return snapshot;
  return {
    state: next,
    previousState: snapshot.state,
    transitionCount: snapshot.transitionCount + 1,
  };
}

/** Réduit un événement FSM — pur, déterministe. */
export function reduceCockpitFsm(
  snapshot: CockpitFsmSnapshot,
  event: CockpitFsmEvent
): CockpitFsmSnapshot {
  if (event.type === "ENTER") {
    return transition(snapshot, event.target);
  }

  switch (event.type) {
    case "DRIVER_SELECT":
      return transition(snapshot, "DRIVER_FOCUS");
    case "DRIVER_CLEAR":
      return snapshot.state === "INCIDENT_ALERT"
        ? snapshot
        : transition(snapshot, "GLOBAL_TRACKING");
    case "SEARCH_OPEN":
      return transition(snapshot, "SEARCH_ACTIVE");
    case "SEARCH_CLOSE":
      return transition(snapshot, "GLOBAL_IDLE");
    case "INCIDENT_RAISE":
      return transition(snapshot, "INCIDENT_ALERT");
    case "INCIDENT_ACK":
      return transition(snapshot, snapshot.previousState ?? "GLOBAL_IDLE");
    case "DISPATCH_OPEN":
      return transition(snapshot, "DISPATCH_ACTION");
    case "DISPATCH_CLOSE":
      return transition(snapshot, "GLOBAL_IDLE");
    case "TRACKING_START":
      if (snapshot.state === "GLOBAL_IDLE") {
        return transition(snapshot, "GLOBAL_TRACKING");
      }
      return snapshot;
    default:
      return snapshot;
  }
}

/** Infère l'état FSM depuis signaux runtime (sans side effects). */
export function inferFsmStateFromSignals(signals: {
  searchActive: boolean;
  selectedDriverId: number | null;
  driverSheetOpen: boolean;
  filtersOpen: boolean;
  layersOpen: boolean;
  urgentCount: number;
}): CockpitFsmState {
  if (signals.urgentCount > 0 && !signals.driverSheetOpen) {
    return "INCIDENT_ALERT";
  }
  if (signals.filtersOpen || signals.layersOpen) {
    return "DISPATCH_ACTION";
  }
  if (signals.searchActive) {
    return "SEARCH_ACTIVE";
  }
  if (signals.selectedDriverId != null && signals.driverSheetOpen) {
    return "DRIVER_FOCUS";
  }
  if (signals.selectedDriverId != null) {
    return "GLOBAL_TRACKING";
  }
  return "GLOBAL_IDLE";
}
