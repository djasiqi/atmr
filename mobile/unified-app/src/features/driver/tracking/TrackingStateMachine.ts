export type TrackingFsmState =
  | "IDLE"
  | "PRESENCE"
  | "MISSION_PREPARE"
  | "MISSION_ACTIVE"
  | "MISSION_BACKGROUND"
  | "MISSION_RECOVERING"
  | "DEGRADED"
  | "MISSION_STOPPING";

export type TrackingFsmInput = {
  hasMission: boolean;
  presenceWindow: boolean;
  appForeground: boolean;
  missionLive: boolean;
  fixStale: boolean;
  circuitOpen: boolean;
  missionTerminal: boolean;
};

export function resolveTrackingFsmState(input: TrackingFsmInput): TrackingFsmState {
  if (input.missionTerminal || (!input.hasMission && !input.presenceWindow)) {
    return "IDLE";
  }
  if (input.circuitOpen || input.fixStale) {
    if (input.missionLive || input.hasMission) {
      return "MISSION_RECOVERING";
    }
    return "DEGRADED";
  }
  if (input.missionLive && input.hasMission) {
    return input.appForeground ? "MISSION_ACTIVE" : "MISSION_BACKGROUND";
  }
  if (input.hasMission && !input.missionLive) {
    return "MISSION_PREPARE";
  }
  if (input.presenceWindow && input.appForeground) {
    return "PRESENCE";
  }
  return "IDLE";
}

export function isRecoveringOrDegraded(fsm: TrackingFsmState): boolean {
  return fsm === "MISSION_RECOVERING" || fsm === "DEGRADED";
}
