/** P1 : entrée du resolver (resolvePresenceState / resolveLocationModeFromState) dans trackingPolicy. */

export type DriverPresenceState =
  | "LOGGED_OUT"
  | "NO_PERMISSION"
  | "OFFLINE"
  | "ONLINE_AVAILABLE"
  | "ONLINE_ON_MISSION"
  | "BACKGROUND_AVAILABLE"
  | "BACKGROUND_ON_MISSION";

export type PresenceInputs = {
  isAuthenticated: boolean;
  isDriver: boolean;
  hasFgPermission: boolean;
  hasBgPermission: boolean;
  appInBackground: boolean;
  hasActiveMission: boolean;
  availabilityPresenceEnabled: boolean;
};

export function resolvePresenceState(inputs: PresenceInputs): DriverPresenceState {
  if (!inputs.isAuthenticated) return "LOGGED_OUT";
  if (!inputs.isDriver) return "OFFLINE";
  if (!inputs.hasFgPermission) return "NO_PERMISSION";
  if (inputs.appInBackground && !inputs.hasBgPermission) return "NO_PERMISSION";

  if (inputs.appInBackground && inputs.hasActiveMission) return "BACKGROUND_ON_MISSION";
  if (inputs.appInBackground && inputs.availabilityPresenceEnabled) {
    return "BACKGROUND_AVAILABLE";
  }
  if (inputs.hasActiveMission) return "ONLINE_ON_MISSION";
  return "ONLINE_AVAILABLE";
}

export function resolveLocationModeFromState(
  state: DriverPresenceState
): "mission_live" | "availability_presence" | "passive_last_known" {
  if (state === "ONLINE_ON_MISSION" || state === "BACKGROUND_ON_MISSION") {
    return "mission_live";
  }
  if (state === "ONLINE_AVAILABLE" || state === "BACKGROUND_AVAILABLE") {
    return "availability_presence";
  }
  return "passive_last_known";
}
