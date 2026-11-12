import {
  cashCollectionSettled,
  dashboardAmountLabel,
  dashboardCashAuthorized,
  billingFollowupLabel,
  cashSheetClosedLine,
  dashboardDestinationStepAuthorized,
  exceptionalCashOffer,
  dashboardPrimaryKind,
  invoiceConfirmationRequired,
  dashboardSettlementAuthorized,
  dashboardSettlementModes,
  dashboardVerificationRequired,
  v1LocalCashAuthorized,
  handoverReadClosed,
  invoicePreparationStarted,
  settlementImpossibleNotice,
  settlementRubricClosed,
  verificationNotice,
} from "./dashboardHandover";

const cash = {
  kind: "collection_authorized",
  reason: "simulation",
  modes: ["cash"],
  simulation: true,
  source: "simulation",
  payment_executable: false,
  persists_payment: false,
  writes_cash_receipt: false,
  writes_invoice: false,
  writes_completed_payment: false,
  writes_financial_signature: false,
  authorization_durable: false,
  stale: false,
};

describe("autorisation espèces du Dashboard", () => {
  it("autorise les espèces simulées", () => {
    expect(dashboardCashAuthorized(cash)).toBe(true);
    expect(dashboardCashAuthorized({ ...cash, modes: ["tap_to_pay", "cash", "invoice"] })).toBe(true);
  });

  it("masque l'institution, le déjà payé, la facture et l'aller-retour sans espèces", () => {
    expect(dashboardCashAuthorized({ ...cash, kind: "collection_forbidden", reason: "institution_payer", modes: [] })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, kind: "already_paid", reason: "payment_confirmed", modes: [] })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, kind: "invoice_authorized", modes: ["invoice"] })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, kind: "collection_forbidden", reason: "aller_retour_bloque", modes: [] })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, modes: ["tap_to_pay"] })).toBe(false);
  });

  it("ouvre la facture simulée sans activer Tap to Pay", () => {
    const invoice = { ...cash, kind: "invoice_authorized", modes: ["invoice"] };
    expect(dashboardSettlementAuthorized(invoice)).toBe(true);
    expect(dashboardDestinationStepAuthorized(invoice)).toBe(true);
    expect(invoiceConfirmationRequired(invoice)).toBe(true);
    const covered = {
      ...invoice,
      billing_followup: {
        invoice_confirmation_required: false,
        label: "Mandat de démonstration applicable. Dossier préparé, facture non émise.",
        real_payment: false,
        invoice_issued: false,
      },
    };
    expect(invoiceConfirmationRequired(covered)).toBe(false);
    expect(dashboardDestinationStepAuthorized(covered)).toBe(false);
    expect(billingFollowupLabel(covered)).toContain("dossier non préparé");
    expect(billingFollowupLabel(covered)).not.toContain("Dossier préparé");
    const persisted = {
      ...covered,
      billing_followup: {
        ...covered.billing_followup,
        dossier_state: "prepared",
      },
    };
    expect(billingFollowupLabel(persisted)).toContain("Dossier préparé");
    expect(billingFollowupLabel(invoice)).toBeNull();
    expect(dashboardCashAuthorized(invoice)).toBe(false);
    expect(dashboardDestinationStepAuthorized({ ...cash, modes: ["cash", "invoice"] })).toBe(true);
    expect(dashboardSettlementAuthorized({ ...cash, modes: ["tap_to_pay"] })).toBe(false);
    expect(dashboardSettlementAuthorized({ ...invoice, writes_invoice: true })).toBe(false);
    expect(dashboardAmountLabel({ ...cash, amount_chf: "45.00" })).toBe("45.00 CHF");
    expect(dashboardAmountLabel({ ...cash, amount_chf: null })).toBe("Montant de la réservation");
    const exceptional = {
      ...invoice,
      exceptional_cash: { allowed: true, label: "Encaisser en espèces" },
    };
    expect(exceptionalCashOffer(exceptional)).toBe("Encaisser en espèces");
    expect(dashboardDestinationStepAuthorized(exceptional)).toBe(true);
    expect(dashboardSettlementModes(exceptional).cash).toBe(false);
    expect(exceptionalCashOffer({ ...cash, exceptional_cash: { allowed: true } })).toBeNull();
  });

  it("refuse une lecture réelle, périmée ou qui prétend encaisser", () => {
    expect(dashboardCashAuthorized({ ...cash, simulation: false, source: "backend", modes: ["cash"] })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, stale: true })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, payment_executable: true })).toBe(false);
    expect(dashboardCashAuthorized({ ...cash, writes_invoice: true })).toBe(false);
    expect(dashboardCashAuthorized(null)).toBe(false);
  });

  it("distingue une rubrique fermée d'une panne", () => {
    expect(
      handoverReadClosed({ response: { status: 404, data: { error: "Rubrique indisponible" } } })
    ).toBe(true);
    expect(
      handoverReadClosed({ response: { status: 404, data: { error: "Mission indisponible" } } })
    ).toBe(true);
    expect(handoverReadClosed({ response: { status: 503, data: { error: "Lecture indisponible" } } })).toBe(false);
    expect(handoverReadClosed(new Error("réseau"))).toBe(false);
    expect(handoverReadClosed({ response: { status: 404, data: { error: "introuvable" } } })).toBe(false);
  });

  it("aligne l'action principale sur l'étape, puis rend Terminer", () => {
    const base = {
      nextTransition: "COMPLETED",
      destinationAuthorized: true,
      destinationConfirmed: false,
      settlementSettled: false,
    };
    expect(dashboardPrimaryKind(base)).toBe("destination");
    expect(dashboardPrimaryKind({ ...base, destinationConfirmed: true })).toBe("reopen");
    expect(dashboardPrimaryKind({ ...base, settlementSettled: true })).toBe("transition");
    expect(dashboardPrimaryKind({ ...base, destinationAuthorized: false })).toBe("transition");
    expect(dashboardPrimaryKind({ ...base, nextTransition: "IN_PROGRESS" })).toBe("transition");
    expect(cashCollectionSettled({ state: "prepared" })).toBe(false);
    expect(cashCollectionSettled({ state: "administrative" })).toBe(true);
    expect(cashCollectionSettled({ state: "uncertain" })).toBe(true);
    expect(invoicePreparationStarted({ id: 4 })).toBe(true);
    expect(invoicePreparationStarted({})).toBe(false);
  });

  it("décide le nombre d'étapes selon le contrat serveur", () => {
    const institution = {
      ...cash,
      kind: "collection_forbidden",
      reason: "institution_payer",
      modes: [],
      billed_to_type: "clinic",
    };
    const mandate = {
      ...cash,
      kind: "invoice_authorized",
      modes: ["invoice"],
      billing_followup: { invoice_confirmation_required: false, label: "Mandat applicable" },
    };
    const invoiceOnly = { ...cash, kind: "invoice_authorized", modes: ["invoice"] };
    const unconfirmed = {
      kind: "financial_situation_unconfirmed",
      source: "booking_only",
      verification_required: true,
      billed_to_type: "patient",
      amount_chf: "45.00",
      modes: [],
      simulation: false,
    };
    expect(dashboardDestinationStepAuthorized(institution)).toBe(false);
    expect(dashboardVerificationRequired(institution)).toBe(false);
    expect(dashboardDestinationStepAuthorized(mandate)).toBe(false);
    expect(dashboardDestinationStepAuthorized(invoiceOnly)).toBe(true);
    expect(dashboardCashAuthorized(invoiceOnly)).toBe(false);
    expect(dashboardDestinationStepAuthorized(cash)).toBe(true);
    expect(dashboardDestinationStepAuthorized(unconfirmed)).toBe(false);
    expect(dashboardVerificationRequired(unconfirmed)).toBe(true);
    expect(
      dashboardVerificationRequired({
        kind: "administrative_review",
        reason: "settlement_policy_missing",
        modes: [],
        simulation: false,
      })
    ).toBe(true);
    expect(dashboardDestinationStepAuthorized({
      kind: "administrative_review",
      reason: "settlement_policy_missing",
      modes: [],
      simulation: false,
    })).toBe(false);
    expect(dashboardSettlementModes(unconfirmed)).toEqual({ cash: false, invoice: false });
    expect(verificationNotice(unconfirmed)).toContain("Vérification du règlement nécessaire");
    expect(verificationNotice(unconfirmed)).toContain("Patient");
    expect(verificationNotice(unconfirmed)).toContain("45.00 CHF");
    expect(settlementRubricClosed({ response: { data: { error: "Rubrique indisponible" } } })).toBe(true);
    expect(settlementRubricClosed({ response: { data: { error: "Mission indisponible" } } })).toBe(false);
  });

  it("affiche le tarif V1 sans ouvrir la feuille espèces", () => {
    const patient = {
      kind: "administrative_review",
      reason: "settlement_policy_missing",
      modes: [],
      simulation: false,
      source: "backend",
      payment_executable: false,
      billed_to_type: "patient",
      amount_chf: null,
      billing_followup: { due_chf: "45.00", balance_chf: "45.00", real_payment: false },
      cash_sheet: {
        allowed: false,
        reason: "especes_non_autorisees",
        real_collection: false,
        simulation: false,
        register: "cash_v1",
      },
    };
    expect(dashboardAmountLabel(patient)).toBe("45.00 CHF");
    expect(dashboardCashAuthorized(patient)).toBe(false);
    expect(dashboardSettlementModes(patient)).toEqual({ cash: false, invoice: false });
    expect(dashboardDestinationStepAuthorized(patient)).toBe(false);
    expect(cashSheetClosedLine(patient)).toBe("Feuille espèces refusée (especes_non_autorisees).");
    expect(
      cashSheetClosedLine({
        kind: "collection_forbidden",
        cash_sheet: { allowed: false, reason: "facture_emise", register: "cash_v1" },
      }),
    ).toBe("Nouvelle collecte interdite (facture_emise).");
    expect(verificationNotice(patient)).toContain("45.00 CHF");
    expect(verificationNotice(patient)).toContain("especes_non_autorisees");
    expect(exceptionalCashOffer(patient)).toBeNull();
    expect(v1LocalCashAuthorized(patient)).toBe(false);
  });

  it("ouvre les espèces V1 locales sans simulation", () => {
    const local = {
      kind: "administrative_review",
      reason: "settlement_policy_missing",
      modes: [],
      simulation: false,
      source: "backend",
      payment_executable: false,
      persists_payment: false,
      billed_to_type: "patient",
      amount_chf: "45.00",
      cash_sheet: {
        allowed: true,
        reason: null,
        real_collection: false,
        simulation: false,
        register: "cash_v1",
        local_ledger: true,
      },
    };
    expect(v1LocalCashAuthorized(local)).toBe(true);
    expect(dashboardSettlementModes(local)).toEqual({ cash: true, invoice: false });
    expect(dashboardCashAuthorized(local)).toBe(true);
    expect(dashboardVerificationRequired(local)).toBe(false);
    expect(dashboardDestinationStepAuthorized(local)).toBe(true);
    expect(cashSheetClosedLine(local)).toBeNull();
    expect(v1LocalCashAuthorized({ ...local, payment_executable: true })).toBe(false);
    expect(
      v1LocalCashAuthorized({
        ...local,
        cash_sheet: { ...local.cash_sheet, simulation: true, register: "simulation" },
      })
    ).toBe(false);
  });

  it("n'annonce un signalement conservé qu'après confirmation serveur", () => {
    expect(settlementImpossibleNotice("sent")).toContain("Signalement conservé");
    expect(settlementImpossibleNotice("queued")).toContain("Signalement non enregistré");
    expect(settlementImpossibleNotice("queued")).not.toContain("Signalement conservé");
    expect(settlementImpossibleNotice("rejected")).toContain("Signalement non enregistré");
    expect(settlementImpossibleNotice("rejected")).toContain("ne sera pas renvoyé");
    expect(settlementImpossibleNotice("rejected")).not.toContain("Signalement conservé");
    expect(settlementImpossibleNotice("sent")).toContain("Aucune déclaration financière");
    expect(settlementImpossibleNotice("queued")).toContain("pourra être repris");
    expect(settlementImpossibleNotice("queued")).toContain("La mission peut être terminée");
  });
});
