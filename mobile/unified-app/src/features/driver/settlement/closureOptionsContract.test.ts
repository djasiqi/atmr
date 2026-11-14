import {
  buildDriverClosureOptions,
  incidentRequiresObservation,
} from "./closureOptionsContract";

describe("options de clôture chauffeur", () => {
  it("propose le différé seulement si convention applicable", () => {
    const withConvention = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "applicable_demonstration",
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
    });
    expect(withConvention.find((o) => o.code === "deferred_convention")?.available).toBe(
      true
    );

    const without = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "absent",
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
    });
    expect(without.find((o) => o.code === "deferred_convention")?.available).toBe(false);
    expect(without.find((o) => o.code === "cash")?.available).toBe(true);
    expect(without.find((o) => o.code === "no_payment_incident")?.available).toBe(true);
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
  });

  it("carte non opérationnelle et observation obligatoire pour autre", () => {
    const menu = buildDriverClosureOptions({
      billedToType: "patient",
      mandateKind: "absent",
      driverCompanyId: 1,
      companyId: 1,
      paymentReceived: false,
      cardProviderOperational: false,
    });
    expect(menu.find((o) => o.code === "card")?.available).toBe(false);
    expect(incidentRequiresObservation("autre", "")).toBe(false);
    expect(incidentRequiresObservation("autre", "Motif détaillé")).toBe(true);
  });
});
