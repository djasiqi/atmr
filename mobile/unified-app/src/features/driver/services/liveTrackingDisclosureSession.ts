/**
 * Flag session : l'utilisateur a accepté la disclosure avant demande permission BG.
 * Évite requestBackgroundPermissionsAsync hors contexte (review Apple / Google).
 */
let disclosureAccepted = false;

export function isLiveTrackingDisclosureAccepted(): boolean {
  return disclosureAccepted;
}

export function markLiveTrackingDisclosureAccepted(): void {
  disclosureAccepted = true;
}

export function resetLiveTrackingDisclosureSession(): void {
  disclosureAccepted = false;
}

/** Test-only */
export function __resetLiveTrackingDisclosureSessionForTests(): void {
  disclosureAccepted = false;
}
