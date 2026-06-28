import {
  __resetDriverDisclosureOrchestratorForTests,
  getDriverDisclosureOrchestrationSnapshot,
  markDriverNotificationDisclosureDismissedSession,
  setDriverDisclosureNotificationAccepted,
  setDriverMissionDisclosureVisible,
  setDriverPresenceDisclosureUiState,
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
});
