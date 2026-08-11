/**
 * Résolveur unique d’éligibilité GPS chauffeur.
 *
 * Signaux séparés (ne pas fusionner) :
 * - driverAvailable / presenceWindowOpen / appForeground /
 *   presenceDisclosureAccepted / hasActiveMission
 */

export type TrackingEligibilityInput = {
  driverAvailable: boolean;
  presenceWindowOpen: boolean;
  appForeground: boolean;
  presenceDisclosureAccepted: boolean;
  hasActiveMission: boolean;
};

export type TrackingEligibilityMode =
  | "OFF"
  | "PRESENCE_FG"
  | "PRESENCE_BG"
  | "MISSION";

export type TrackingEligibilityResult = {
  missionEligible: boolean;
  foregroundPresenceEligible: boolean;
  backgroundPresenceEligible: boolean;
  trackingEligible: boolean;
  mode: TrackingEligibilityMode;
};

export function resolveTrackingEligibility(
  input: TrackingEligibilityInput
): TrackingEligibilityResult {
  const missionEligible = Boolean(input.hasActiveMission);
  const disclosure = Boolean(input.presenceDisclosureAccepted);
  const available = Boolean(input.driverAvailable);
  const foreground = Boolean(input.appForeground);
  const windowOpen = Boolean(input.presenceWindowOpen);

  // P0-F TIME : présence FG aussi bornée par la fenêtre 07–19 Europe/Zurich
  const foregroundPresenceEligible =
    available && foreground && disclosure && windowOpen;
  const backgroundPresenceEligible =
    available && !foreground && windowOpen && disclosure;

  const trackingEligible =
    missionEligible || foregroundPresenceEligible || backgroundPresenceEligible;

  let mode: TrackingEligibilityMode = "OFF";
  if (missionEligible) {
    mode = "MISSION";
  } else if (foregroundPresenceEligible) {
    mode = "PRESENCE_FG";
  } else if (backgroundPresenceEligible) {
    mode = "PRESENCE_BG";
  }

  return {
    missionEligible,
    foregroundPresenceEligible,
    backgroundPresenceEligible,
    trackingEligible,
    mode,
  };
}

/** Accuracy GPS selon le mode d’éligibilité (Lot B). */
export type PresenceGpsAccuracy = "high" | "balanced";

export function resolvePresenceGpsAccuracy(input: {
  hasActiveMission: boolean;
  appForeground: boolean;
}): PresenceGpsAccuracy {
  if (input.hasActiveMission || input.appForeground) {
    return "high";
  }
  return "balanced";
}
