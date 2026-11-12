import { settlementHttpNotice } from "./settlementHttpNotice";

const lost = "État incertain. Aucune seconde réception automatique.";

describe("avis HTTP du règlement", () => {
  it("garde l'incertitude quand aucune réponse n'arrive", () => {
    expect(settlementHttpNotice(new Error("réseau"), lost)).toBe(lost);
  });

  it("affiche le refus métier sans le confondre avec une perte", () => {
    const error = { response: { status: 409, data: { error_code: "deja_paye" } } };
    expect(settlementHttpNotice(error, lost)).toBe("Refus : cette course est déjà payée.");
  });

  it("explique un dépassement du solde encore encaissable", () => {
    const error = { response: { status: 409, data: { error_code: "depassement_autorise" } } };
    expect(settlementHttpNotice(error, lost)).toBe(
      "Refus : le montant dépasse le solde encore encaissable."
    );
  });
});
