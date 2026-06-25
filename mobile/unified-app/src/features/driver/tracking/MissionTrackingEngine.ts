import type { DriverTrackingMode } from "../services/driverTrackingQueue";

/** Moteur mission — FGS + cadence agressive. */
export function resolveMissionTrackingModeValue(): DriverTrackingMode {
  return "mission_live";
}

export function isMissionEngineEligible(input: {
  hasActiveMission: boolean;
  missionLive: boolean;
}): boolean {
  return input.hasActiveMission && input.missionLive;
}
