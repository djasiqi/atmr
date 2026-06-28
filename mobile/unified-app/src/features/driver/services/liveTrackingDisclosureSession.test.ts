import {
  clearPresenceDisclosureDeclined,
  isPresenceDisclosureDeclined,
  markPresenceDisclosureDeclined,
  __resetLiveTrackingDisclosureSessionForTests,
} from "./liveTrackingDisclosureSession";

describe("liveTrackingDisclosureSession", () => {
  beforeEach(() => {
    __resetLiveTrackingDisclosureSessionForTests();
  });

  it("clearPresenceDisclosureDeclined réactive la modale après un refus", () => {
    markPresenceDisclosureDeclined();
    expect(isPresenceDisclosureDeclined()).toBe(true);
    clearPresenceDisclosureDeclined();
    expect(isPresenceDisclosureDeclined()).toBe(false);
  });
});
