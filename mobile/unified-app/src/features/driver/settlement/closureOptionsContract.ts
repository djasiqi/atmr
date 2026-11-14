/**
 * Contrat d'options de clôture — aligné sur le domaine backend CR-02.
 * Terminal externe ≠ Tap to Pay. Convention démo ≠ différé réel.
 */

export type ClosureOptionCode =
  | "deferred_convention"
  | "cash"
  | "card_external_terminal"
  | "card_tap_to_pay"
  | "no_payment_incident";

export type SettlementIncidentMotif =
  | "refus_signer_et_payer"
  | "impossibilite_regler"
  | "contestation_montant"
  | "moyen_paiement_indisponible_ou_echoue"
  | "autre";

export type ClosureOptionView = {
  code: ClosureOptionCode;
  label: string;
  available: boolean;
  recordsPayment: boolean;
  authorizesDeferredBilling: boolean;
  opensIncident: boolean;
  reason?: string;
  channel?: string;
};

export const SETTLEMENT_INCIDENT_MOTIFS: {
  code: SettlementIncidentMotif;
  label: string;
}[] = [
  { code: "refus_signer_et_payer", label: "Refus de signer et de payer" },
  { code: "impossibilite_regler", label: "Impossibilité de régler" },
  { code: "contestation_montant", label: "Contestation du montant" },
  {
    code: "moyen_paiement_indisponible_ou_echoue",
    label: "Moyen de paiement indisponible ou échoué",
  },
  { code: "autre", label: "Autre motif (observation obligatoire)" },
];

type MenuInput = {
  billedToType: string;
  mandateKind: string;
  driverCompanyId: number;
  companyId: number;
  paymentReceived: boolean;
  conventionLegallyEffective?: boolean;
  tapToPayOperational?: boolean;
  /** @deprecated CR-01 alias → tapToPayOperational */
  cardProviderOperational?: boolean;
};

export function buildDriverClosureOptions(input: MenuInput): ClosureOptionView[] {
  if (input.driverCompanyId !== input.companyId) {
    return [];
  }
  const payer = (input.billedToType || "").toLowerCase();
  if (payer === "clinic" || payer === "institution") {
    return [];
  }
  if (input.paymentReceived) {
    return [];
  }

  const legal = input.conventionLegallyEffective === true;
  const realDeferred = input.mandateKind === "applicable" && legal;
  const demoKind =
    input.mandateKind === "applicable_demonstration" ||
    input.mandateKind === "specimen";
  const tapOk =
    input.tapToPayOperational === true || input.cardProviderOperational === true;

  const options: ClosureOptionView[] = [];

  if (realDeferred) {
    options.push({
      code: "deferred_convention",
      label: "Facturation différée autorisée par convention",
      available: true,
      recordsPayment: false,
      authorizesDeferredBilling: true,
      opensIncident: false,
    });
  } else if (demoKind) {
    options.push({
      code: "deferred_convention",
      label: "Facturation différée (convention de démonstration)",
      available: true,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: false,
      reason: "demo_convention_not_legally_effective",
    });
  } else {
    options.push({
      code: "deferred_convention",
      label: "Facturation différée",
      available: false,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: false,
      reason: "convention_not_applicable",
    });
  }

  options.push(
    {
      code: "cash",
      label: "Paiement espèces",
      available: true,
      recordsPayment: true,
      authorizesDeferredBilling: false,
      opensIncident: false,
      channel: "cash",
    },
    {
      code: "card_external_terminal",
      label: "Paiement carte (terminal bancaire externe)",
      available: true,
      recordsPayment: true,
      authorizesDeferredBilling: false,
      opensIncident: false,
      channel: "external_terminal",
    },
    {
      code: "card_tap_to_pay",
      label: "Paiement carte (Tap to Pay LIRIE)",
      available: tapOk,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: false,
      reason: tapOk ? undefined : "tap_to_pay_not_operational",
      channel: "tap_to_pay",
    },
    {
      code: "no_payment_incident",
      label: "Aucun règlement — signalement",
      available: true,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: true,
    }
  );

  return options;
}

export function incidentRequiresObservation(
  motif: SettlementIncidentMotif,
  observation: string
): boolean {
  if (motif !== "autre") {
    return true;
  }
  return observation.trim().length > 0;
}

export function externalCardProofValid(reference: string, receiptNote: string): boolean {
  return reference.trim().length > 0 || receiptNote.trim().length > 0;
}
