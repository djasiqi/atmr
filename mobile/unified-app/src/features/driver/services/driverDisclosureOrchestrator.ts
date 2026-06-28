/**
 * Orchestration des disclosures chauffeur : une modale in-app à la fois au login.
 * Priorité : notifications → disponibilité flotte.
 */
import {
  readNotificationDisclosureAccepted,
  subscribeNotificationDisclosureAccepted,
} from "../../../core/notifications/notificationDisclosurePersistence";

export type DriverDisclosureOrchestrationSnapshot = {
  notificationAccepted: boolean;
  notificationDismissedSession: boolean;
  /** La modale notifications doit être résolue avant la disclosure flotte */
  blocksPresenceDisclosure: boolean;
  presenceHintVisible: boolean;
  presenceModalVisible: boolean;
  missionDisclosureVisible: boolean;
  /** Masquer la bannière rouge tracking (P4) */
  suppressTrackingBanner: boolean;
};

let notificationAccepted = false;
let notificationDismissedSession = false;
let presenceHintVisible = false;
let presenceModalVisible = false;
let missionDisclosureVisible = false;

const listeners = new Set<() => void>();

function notify(): void {
  listeners.forEach((listener) => listener());
}

function buildSnapshot(): DriverDisclosureOrchestrationSnapshot {
  const blocksPresenceDisclosure =
    !notificationAccepted && !notificationDismissedSession;
  return {
    notificationAccepted,
    notificationDismissedSession,
    blocksPresenceDisclosure,
    presenceHintVisible,
    presenceModalVisible,
    missionDisclosureVisible,
    suppressTrackingBanner:
      presenceHintVisible ||
      presenceModalVisible ||
      blocksPresenceDisclosure ||
      missionDisclosureVisible,
  };
}

export function getDriverDisclosureOrchestrationSnapshot(): DriverDisclosureOrchestrationSnapshot {
  return buildSnapshot();
}

export function subscribeDriverDisclosureOrchestration(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function setDriverDisclosureNotificationAccepted(accepted: boolean): void {
  notificationAccepted = accepted;
  notify();
}

export function markDriverNotificationDisclosureDismissedSession(): void {
  notificationDismissedSession = true;
  notify();
}

export function setDriverPresenceDisclosureUiState(
  partial: Partial<Pick<DriverDisclosureOrchestrationSnapshot, "presenceHintVisible" | "presenceModalVisible">>
): void {
  if (partial.presenceHintVisible !== undefined) {
    presenceHintVisible = partial.presenceHintVisible;
  }
  if (partial.presenceModalVisible !== undefined) {
    presenceModalVisible = partial.presenceModalVisible;
  }
  notify();
}

export function setDriverMissionDisclosureVisible(visible: boolean): void {
  missionDisclosureVisible = visible;
  notify();
}

export function initDriverDisclosureOrchestration(): () => void {
  const syncAccepted = async () => {
    const accepted = await readNotificationDisclosureAccepted();
    setDriverDisclosureNotificationAccepted(accepted);
  };
  void syncAccepted();
  const unsubscribe = subscribeNotificationDisclosureAccepted(() => {
    void syncAccepted();
  });
  return unsubscribe;
}

/** Test-only */
export function __resetDriverDisclosureOrchestratorForTests(): void {
  notificationAccepted = false;
  notificationDismissedSession = false;
  presenceHintVisible = false;
  presenceModalVisible = false;
  missionDisclosureVisible = false;
  notify();
}
