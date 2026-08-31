/**
 * Résolveur unique d’éligibilité GPS chauffeur (contrat produit v4).
 *
 * Signaux séparés (ne pas fusionner) :
 * - driverAvailable (SoT Driver.is_available ; null = UNKNOWN, pas encore hydraté)
 * - appForeground
 * - presenceDisclosureAccepted / permissionsReady
 * - hasActiveMission
 *
 * La fenêtre horaire 07–19 n’est plus une gate produit.
 */

export type TrackingEligibilityInput = {
  /** true = en service ; false = hors service ; null/undefined = UNKNOWN (pas encore hydraté). */
  driverAvailable: boolean | null;
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
  /** En service mais contrat non garanti (permissions / disclosure). */
  blocked: boolean;
  /** SoT pas encore hydratée : pas PRESENCE/LIVE, pas hors service. */
  availabilityPending: boolean;
  mode: TrackingEligibilityMode;
};

export function resolveTrackingEligibility(
  input: TrackingEligibilityInput
): TrackingEligibilityResult {
  const availabilityPending = input.driverAvailable == null;
  const available = input.driverAvailable === true;
  const foreground = Boolean(input.appForeground);
  const disclosure = Boolean(input.presenceDisclosureAccepted);
  const permissionsReady =
    input.permissionsReady !== undefined
      ? Boolean(input.permissionsReady)
      : disclosure;
  const capabilityReady = permissionsReady && disclosure;
  const hasMission = Boolean(input.hasActiveMission);

  const empty = {
    missionEligible: false,
    foregroundPresenceEligible: false,
    backgroundPresenceEligible: false,
    trackingEligible: false,
    blocked: false,
    availabilityPending,
    mode: "OFF" as TrackingEligibilityMode,
  };

  if (availabilityPending) {
    return empty;
  }

  if (!available) {
    return empty;
  }

  // En service sans capacité de garantir le contrat → BLOCKED (≠ OFF), mission comprise.
  if (!capabilityReady) {
    return {
      ...empty,
      blocked: true,
      mode: "BLOCKED",
    };
  }

  const missionEligible = hasMission;
  const foregroundPresenceEligible = !hasMission && foreground;
  const backgroundPresenceEligible = !hasMission && !foreground;
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
    blocked: false,
    availabilityPending: false,
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
