import { realtimeSocketDotColor, realtimeStatusA11yLabel } from "./companyRealtimeFluxStatus";

describe("companyRealtimeFluxStatus", () => {
  it("pastille rouge et libellé déconnecté en idle", () => {
    expect(realtimeSocketDotColor("idle")).toBe("#dc3545");
    expect(realtimeStatusA11yLabel("idle")).toBe("Flux temps réel déconnecté");
  });

  it("pastille verte en healthy", () => {
    expect(realtimeSocketDotColor("healthy")).toBe("#00796B");
    expect(realtimeStatusA11yLabel("healthy")).toBe("Flux temps réel connecté");
  });
});
