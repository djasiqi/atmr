import {
  __resetDriverDisclosureOrchestratorForTests,
  getDriverDisclosureOrchestrationSnapshot,
  markDriverNotificationDisclosureDismissedSession,
  setDriverDisclosureNotificationAccepted,
  setDriverMissionDisclosureVisible,
  setDriverPresenceDisclosureUiState,
  setDriverTrackingReadinessPanelVisible,
} from "./driverDisclosureOrchestrator";

describe("driverDisclosureOrchestrator", () => {
  beforeEach(() => {
    __resetDriverDisclosureOrchestratorForTests();
  });

  it("bloque la disclosure flotte tant que notifications non résolues", () => {
    expect(getDriverDisclosureOrchestrationSnapshot().blocksPresenceDisclosure).toBe(true);

    markDriverNotificationDisclosureDismissedSession();
    expect(getDriverDisclosureOrchestrationSnapshot().blocksPresenceDisclosure).toBe(false);
  });

  it("libère la disclosure flotte après acceptation notifications", () => {
    setDriverDisclosureNotificationAccepted(true);
    expect(getDriverDisclosureOrchestrationSnapshot().blocksPresenceDisclosure).toBe(false);
  });

  it("masque la bannière tracking quand présence ou mission en attente (P4)", () => {
    setDriverPresenceDisclosureUiState({ presenceHintVisible: true });
    expect(getDriverDisclosureOrchestrationSnapshot().suppressTrackingBanner).toBe(true);

    __resetDriverDisclosureOrchestratorForTests();
    setDriverMissionDisclosureVisible(true);
    expect(getDriverDisclosureOrchestrationSnapshot().suppressTrackingBanner).toBe(true);
  });

  it("signale le panneau préparation tracking pour supprimer les doublons UI", () => {
    setDriverTrackingReadinessPanelVisible(true);
    const snap = getDriverDisclosureOrchestrationSnapshot();
    expect(snap.trackingReadinessPanelVisible).toBe(true);
    expect(snap.suppressTrackingBanner).toBe(true);

    setDriverTrackingReadinessPanelVisible(false);
    expect(getDriverDisclosureOrchestrationSnapshot().trackingReadinessPanelVisible).toBe(false);
  });
});
