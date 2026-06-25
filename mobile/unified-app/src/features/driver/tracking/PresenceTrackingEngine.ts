import type { DriverTrackingMode } from "../services/driverTrackingQueue";

/** Moteur présence — foreground watch uniquement, pas de FGS. */
export function resolvePresenceTrackingMode(): DriverTrackingMode {
  return "availability_presence";
}

export function isPresenceEngineEligible(input: {
  presenceWindowActive: boolean;
  appForeground: boolean;
  hasActiveMission: boolean;
}): boolean {
  return input.presenceWindowActive && input.appForeground && !input.hasActiveMission;
}
