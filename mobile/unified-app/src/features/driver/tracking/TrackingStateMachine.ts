export type TrackingFsmState =
  | "IDLE"
  | "BLOCKED"
  | "PRESENCE"
  | "MISSION_PREPARE"
  | "MISSION_ACTIVE"
  | "MISSION_BACKGROUND"
  | "MISSION_RECOVERING"
  | "DEGRADED"
  | "MISSION_STOPPING";

export type TrackingFsmInput = {
  hasMission: boolean;
  /** Présence FG ou BG éligible (permissionsReady + en_service). */
  presenceEligible: boolean;
  /**
   * En service mais permissions insuffisantes pour garantir le contrat.
   * Distinct de IDLE / hors service.
   */
  blocked?: boolean;
  appForeground: boolean;
  missionLive: boolean;
  fixStale: boolean;
  circuitOpen: boolean;
  missionTerminal: boolean;
  /** En service (Driver.is_available). Requis pour fin mission → PRESENCE. */
  enService?: boolean;
  /**
   * @deprecated Utiliser `presenceEligible`. Conservé pour compat tests/anciens appels.
   */
  presenceWindow?: boolean;
};

export function resolveTrackingFsmState(input: TrackingFsmInput): TrackingFsmState {
  const presenceEligible = input.presenceEligible ?? Boolean(input.presenceWindow);
  const enService = input.enService ?? (presenceEligible || Boolean(input.blocked));
  const blocked = Boolean(input.blocked);

  // Fin de mission seule ≠ OFF : si encore en service → PRESENCE (ou BLOCKED)
  if (input.missionTerminal) {
    if (!enService) {
      return "IDLE";
    }
    if (blocked || !presenceEligible) {
      return "BLOCKED";
    }
    return "PRESENCE";
  }

  if (!input.hasMission && !presenceEligible && !blocked) {
    return "IDLE";
  }

  if (input.circuitOpen || input.fixStale) {
    if (input.missionLive || input.hasMission) {
      return "MISSION_RECOVERING";
    }
    if (blocked) {
      return "BLOCKED";
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
  if (blocked) {
    return "BLOCKED";
  }
  return "IDLE";
}

export function isRecoveringOrDegraded(fsm: TrackingFsmState): boolean {
  return fsm === "MISSION_RECOVERING" || fsm === "DEGRADED";
}
