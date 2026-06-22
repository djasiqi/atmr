import { isCompanyTransportStableForMapOverlays } from "./companyMapNativeOverlayGate";

describe("isCompanyTransportStableForMapOverlays", () => {
  it("accepte uniquement healthy", () => {
    expect(isCompanyTransportStableForMapOverlays("healthy")).toBe(true);
    expect(isCompanyTransportStableForMapOverlays("connecting")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("reconnecting")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("failed")).toBe(false);
    expect(isCompanyTransportStableForMapOverlays("idle")).toBe(false);
  });
});
