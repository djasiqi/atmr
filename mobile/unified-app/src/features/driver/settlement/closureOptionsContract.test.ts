import {
  buildDriverClosureOptions,
  externalCardProofValid,
  incidentRequiresObservation,
} from "./closureOptionsContract";

describe("options de clôture chauffeur CR-02", () => {
  it("convention démo visible sans autoriser le différé réel", () => {
    const withDemo = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "applicable_demonstration",
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
    });
    const deferred = withDemo.find((o) => o.code === "deferred_convention");
    expect(deferred?.available).toBe(true);
    expect(deferred?.authorizesDeferredBilling).toBe(false);

    const real = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "applicable",
      conventionLegallyEffective: true,
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
    });
    expect(
      real.find((o) => o.code === "deferred_convention")?.authorizesDeferredBilling
    ).toBe(true);
  });

  it("distingue terminal externe et Tap to Pay", () => {
    const menu = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "absent",
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
      tapToPayOperational: false,
    });
    expect(menu.find((o) => o.code === "card_external_terminal")?.available).toBe(true);
    expect(menu.find((o) => o.code === "card_tap_to_pay")?.available).toBe(false);
    expect(externalCardProofValid("AUTH-123", "")).toBe(true);
    expect(externalCardProofValid("", "")).toBe(false);
  });

  it("refuse l'accès inter-entreprises et les institutions", () => {
    expect(
      buildDriverClosureOptions({
        billedToType: "patient",
        mandateKind: "absent",
        driverCompanyId: 2,
        companyId: 1,
        paymentReceived: false,
      })
    ).toEqual([]);
    expect(
      buildDriverClosureOptions({
        billedToType: "clinic",
        mandateKind: "absent",
        driverCompanyId: 1,
        companyId: 1,
        paymentReceived: false,
      })
    ).toEqual([]);
    expect(incidentRequiresObservation("autre", "Motif")).toBe(true);
  });
});
