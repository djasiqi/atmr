/**
 * Résolveur unique d’éligibilité GPS chauffeur (contrat produit v4).
 *
 * Signaux séparés (ne pas fusionner) :
 * - driverAvailable (SoT Driver.is_available)
 * - appForeground
 * - presenceDisclosureAccepted / permissionsReady
 * - hasActiveMission
 *
 * La fenêtre horaire 07–19 n’est plus une gate produit.
 */

export type TrackingEligibilityInput = {
  driverAvailable: boolean;
  /** @deprecated Ignoré pour l’éligibilité produit (conservé pour call sites hérités). */
  presenceWindowOpen?: boolean;
  appForeground: boolean;
  presenceDisclosureAccepted: boolean;
  hasActiveMission: boolean;
  /**
   * Capacité de garantir le contrat (FG+BG).
   * Si omis : dérivé de presenceDisclosureAccepted (rétrocompat).
   */
  permissionsReady?: boolean;
};

export type TrackingEligibilityMode =
  | "OFF"
  | "BLOCKED"
  | "PRESENCE_FG"
  | "PRESENCE_BG"
  | "MISSION";

export type TrackingEligibilityResult = {
  missionEligible: boolean;
  foregroundPresenceEligible: boolean;
  backgroundPresenceEligible: boolean;
  trackingEligible: boolean;
  /** En service mais contrat non garanti (permissions). */
  blocked: boolean;
  mode: TrackingEligibilityMode;
};

export function resolveTrackingEligibility(
  input: TrackingEligibilityInput
): TrackingEligibilityResult {
  const missionEligible = Boolean(input.hasActiveMission);
  const available = Boolean(input.driverAvailable);
  const foreground = Boolean(input.appForeground);
  const disclosure = Boolean(input.presenceDisclosureAccepted);
  const permissionsReady =
    input.permissionsReady !== undefined
      ? Boolean(input.permissionsReady)
      : disclosure;

  // En service sans capacité de garantir le contrat → BLOCKED (≠ OFF)
  const blocked = available && !missionEligible && !permissionsReady;

  const foregroundPresenceEligible =
    available && foreground && permissionsReady;
  const backgroundPresenceEligible =
    available && !foreground && permissionsReady;

  const trackingEligible =
    missionEligible || foregroundPresenceEligible || backgroundPresenceEligible;

  let mode: TrackingEligibilityMode = "OFF";
  if (missionEligible) {
    mode = "MISSION";
  } else if (blocked) {
    mode = "BLOCKED";
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
    blocked,
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
