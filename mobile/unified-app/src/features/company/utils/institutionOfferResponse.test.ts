import {
  resolveInstitutionOfferTerminalState,
} from "./institutionOfferResponse";

describe("resolveInstitutionOfferTerminalState", () => {
  it("marque expirée une offre PENDING dont le délai est dépassé", () => {
    expect(
      resolveInstitutionOfferTerminalState(
        {
          status: "PENDING",
          can_respond: false,
          expires_at: "2020-01-01T12:00:00Z",
        },
        new Date("2026-06-22T18:00:00Z")
      )
    ).toBe("expired");
  });

  it("marque indisponible une offre prise par un autre transporteur", () => {
    expect(
      resolveInstitutionOfferTerminalState(
        {
          status: "UNAVAILABLE",
          can_respond: false,
          expires_at: "2030-01-01T12:00:00Z",
        },
        new Date("2026-06-22T18:00:00Z")
      )
    ).toBe("unavailable");
  });
});
