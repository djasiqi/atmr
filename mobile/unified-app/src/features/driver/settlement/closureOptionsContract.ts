/**
 * Contrat d'options de clôture — aligné sur le domaine backend.
 * Aucun paiement fictif, Tap to Pay non opérationnel.
 */

export type ClosureOptionCode =
  | "deferred_convention"
  | "cash"
  | "card"
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

  const conventionOk =
    input.mandateKind === "applicable_demonstration" ||
    input.mandateKind === "applicable";

  const options: ClosureOptionView[] = [
    {
      code: "deferred_convention",
      label: "Facturation différée autorisée par convention",
      available: conventionOk,
      recordsPayment: false,
      authorizesDeferredBilling: conventionOk,
      opensIncident: false,
      reason: conventionOk ? undefined : "convention_not_applicable",
    },
    {
      code: "cash",
      label: "Paiement espèces",
      available: true,
      recordsPayment: true,
      authorizesDeferredBilling: false,
      opensIncident: false,
    },
    {
      code: "card",
      label: "Paiement carte",
      available: input.cardProviderOperational === true,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: false,
      reason:
        input.cardProviderOperational === true
          ? undefined
          : "card_provider_not_operational",
    },
    {
      code: "no_payment_incident",
      label: "Aucun règlement — signalement",
      available: true,
      recordsPayment: false,
      authorizesDeferredBilling: false,
      opensIncident: true,
    },
  ];

  return options.filter((option) => option.available || option.code === "card" || option.code === "deferred_convention");
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
