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
  /** Présence FG ou BG éligible (pas seulement la fenêtre horaire). */
  presenceEligible: boolean;
  appForeground: boolean;
  missionLive: boolean;
  fixStale: boolean;
  circuitOpen: boolean;
  missionTerminal: boolean;
  /**
   * @deprecated Utiliser `presenceEligible`. Conservé pour compat tests/anciens appels.
   */
  presenceWindow?: boolean;
};

export function resolveTrackingFsmState(input: TrackingFsmInput): TrackingFsmState {
  const presenceEligible = input.presenceEligible ?? Boolean(input.presenceWindow);
  if (input.missionTerminal || (!input.hasMission && !presenceEligible)) {
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
  if (presenceEligible) {
    return "PRESENCE";
  }
  return "IDLE";
}

export function isRecoveringOrDegraded(fsm: TrackingFsmState): boolean {
  return fsm === "MISSION_RECOVERING" || fsm === "DEGRADED";
}
