/**
 * Flags session : disclosure mission vs disponibilité flotte avant permission BG.
 */
let missionDisclosureAccepted = false;
let presenceDisclosureAccepted = false;
let presenceDisclosureDeclined = false;

export function isLiveTrackingDisclosureAccepted(): boolean {
  return missionDisclosureAccepted;
}

export function markLiveTrackingDisclosureAccepted(): void {
  missionDisclosureAccepted = true;
}

export function isPresenceDisclosureAccepted(): boolean {
  return presenceDisclosureAccepted;
}

export function markPresenceDisclosureAccepted(): void {
  presenceDisclosureAccepted = true;
  presenceDisclosureDeclined = false;
}

export function isPresenceDisclosureDeclined(): boolean {
  return presenceDisclosureDeclined;
}

export function markPresenceDisclosureDeclined(): void {
  presenceDisclosureDeclined = true;
}

export function resetLiveTrackingDisclosureSession(): void {
  missionDisclosureAccepted = false;
  presenceDisclosureAccepted = false;
  presenceDisclosureDeclined = false;
}

/** Test-only */
export function __resetLiveTrackingDisclosureSessionForTests(): void {
  resetLiveTrackingDisclosureSession();
}
