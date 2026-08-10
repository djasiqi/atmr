import type { DriverTrackingMode } from "../services/driverTrackingQueue";
import {
  resolveTrackingEligibility,
  type TrackingEligibilityInput,
} from "./trackingEligibility";

/** Moteur présence — foreground watch uniquement, pas de FGS. */
export function resolvePresenceTrackingMode(): DriverTrackingMode {
  return "availability_presence";
}

export function isPresenceEngineEligible(
  input: TrackingEligibilityInput
): boolean {
  const result = resolveTrackingEligibility(input);
  return (
    result.foregroundPresenceEligible || result.backgroundPresenceEligible
  );
}
