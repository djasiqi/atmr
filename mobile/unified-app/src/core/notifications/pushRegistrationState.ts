import {
  readNotificationDisclosureAccepted,
  subscribeNotificationDisclosureAccepted,
} from "./notificationDisclosurePersistence";
import { hasPendingPushTokenRegistrations } from "./pendingPushTokenRegistration";
import { getPushPermissionDenied } from "./pushPermissionState";

export type PushRegistrationBannerState =
  | "ok"
  | "disclosure_required"
  | "permission_denied"
  | "registration_pending"
  | "registration_failed";

type Listener = () => void;

const listeners = new Set<Listener>();
let registrationFailed = false;
let disclosureShowRequest = 0;

export function subscribePushRegistrationState(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function notifyListeners(): void {
  listeners.forEach((listener) => listener());
}

export function setPushRegistrationFailed(value: boolean): void {
  if (registrationFailed === value) return;
  registrationFailed = value;
  notifyListeners();
}

export function clearPushRegistrationFailed(): void {
  setPushRegistrationFailed(false);
}

export function requestNotificationDisclosure(): void {
  disclosureShowRequest += 1;
  notifyListeners();
}

export function getDisclosureShowRequestCount(): number {
  return disclosureShowRequest;
}

export async function resolvePushRegistrationBannerState(): Promise<PushRegistrationBannerState> {
  const disclosureAccepted = await readNotificationDisclosureAccepted();
  if (!disclosureAccepted) {
    return "disclosure_required";
  }
  if (getPushPermissionDenied()) {
    return "permission_denied";
  }
  if (await hasPendingPushTokenRegistrations()) {
    return "registration_pending";
  }
  if (registrationFailed) {
    return "registration_failed";
  }
  return "ok";
}

export function subscribePushRegistrationRefresh(onRefresh: () => void): () => void {
  const wrapped = () => onRefresh();
  const unsubDisclosure = subscribeNotificationDisclosureAccepted(wrapped);
  const unsubState = subscribePushRegistrationState(wrapped);
  return () => {
    unsubDisclosure();
    unsubState();
  };
}
