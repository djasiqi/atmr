import {
  representativeComplete,
  SIGNATURE_STEP_LABELS,
  signatureOffer,
  signatureRequest,
  traceAcceptable,
} from "./mandateSignatureFlow";

describe("parcours de signature de démonstration", () => {
  it("demande le parcours quand le mandat est préparé ou absent", () => {
    expect(signatureOffer("prepared", "demo_draft")).toBe("wizard");
    expect(signatureOffer("absent", "demo_absent")).toBe("wizard");
  });

  it("ne redemande pas une signature déjà applicable", () => {
    expect(signatureOffer("applicable_demonstration", "demo_valid_simulated")).toBe("already");
    expect(signatureOffer("prepared", "demo_valid_simulated")).toBe("already");
  });

  it("bloque une couverture expirée, incomplète ou d'une autre entreprise", () => {
    expect(signatureOffer("expired", "demo_draft")).toBe("blocked");
    expect(signatureOffer("amendment_required", "demo_draft")).toBe("blocked");
    expect(signatureOffer("other_company", "demo_draft")).toBe("blocked");
    expect(signatureOffer("not_concerned", "demo_draft")).toBe("blocked");
  });

  it("ouvre le document quand la démonstration est déjà enregistrée", () => {
    expect(signatureOffer("specimen", "demo_specimen_signed")).toBe("document");
  });

  it("refuse un tracé vide et accepte un tracé tactile", () => {
    expect(traceAcceptable([])).toBe(false);
    expect(traceAcceptable([{ x: 12, y: 40 }])).toBe(true);
  });

  it("n'accepte un représentant qu'avec nom, qualité et pouvoir", () => {
    expect(representativeComplete("A", "proche", "fils")).toBe(false);
    expect(representativeComplete("Ada Martin", "proche", "fille")).toBe(true);
  });

  it("signe la version affichée, pas une version plus récente", () => {
    const payload = signatureRequest({
      trace: [{ x: 12, y: 40 }],
      signer: "representative",
      representativeName: "Ada Martin",
      representativeQuality: "proche",
      representativeRelation: "fille",
      tariffVersion: "1",
      versionNumber: 2,
      idempotencyKey: "idemp-signature-version-deux",
    });
    expect(payload.version_number).toBe(2);
    expect(payload.signer).toBe("representative");
    expect(payload.representative_quality).toBe("proche");
    expect(payload.acknowledged).toBe(true);
    expect(signatureRequest({
      trace: [{ x: 1, y: 2 }],
      signer: "patient",
      tariffVersion: "1",
      versionNumber: null,
      idempotencyKey: "idemp-signature-patient-un",
    }).representative_name).toBeNull();
  });

  it("affiche les six étapes dans l'ordre", () => {
    expect(SIGNATURE_STEP_LABELS).toEqual([
      "Identité",
      "Tarifs",
      "Conditions",
      "Signature",
      "Document",
      "Transmission",
    ]);
  });
});
