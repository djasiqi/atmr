const REFUSAL: Record<string, string> = {
  deja_paye: "Refus : cette course est déjà payée.",
  couverture_ambigue: "Refus : la couverture du paiement est ambiguë.",
  facture_emise: "Refus : une facture est déjà émise.",
  institution_facturee: "Refus : le débiteur est une institution.",
  transport_annule: "Refus : la réservation est annulée.",
  chauffeur_non_affecte: "Refus : ce chauffeur n'est pas affecté.",
  deja_encaisse: "Refus : le dossier espèces est déjà confirmé.",
  depassement_autorise: "Refus : le montant dépasse le solde encore encaissable.",
  decision_obsolete: "Refus : le montant serveur a changé.",
  tentative_deja_ouverte: "Refus : une opération est déjà ouverte.",
  aller_retour_bloque: "Refus : l'aller-retour est déjà couvert.",
  statut_incompatible: "Refus : l'état de la mission ne permet pas cette opération.",
  monnaie_non_confirmee: "Refus : la monnaie rendue n'est pas confirmée.",
  rapprochement_reserve: "Refus : le rapprochement est réservé à l'entreprise.",
  emission_desactivee: "Refus : aucune facture réelle n'est émise.",
  signature_reelle_interdite: "Refus : aucune signature réelle n'est collectée.",
  attestation_distincte: "Refus : l'attestation ne vaut pas acceptation.",
  representation_non_approuvee: "Refus : le chauffeur ne vérifie pas le pouvoir.",
  acceptation_non_fabriquee: "Refus : cette action ne crée pas d'acceptation.",
};

type HttpError = {
  response?: { status?: number; data?: { error_code?: string } };
};

/** Une réponse HTTP structurée n'est pas une perte de réseau. */
export function settlementHttpNotice(error: unknown, lostMessage: string): string {
  const response = (error as HttpError | null)?.response;
  if (!response) return lostMessage;
  const code = String(response.data?.error_code || "");
  if (REFUSAL[code]) return REFUSAL[code];
  if (response.status === 401 || response.status === 403) {
    return "Accès refusé. Aucune nouvelle opération.";
  }
  if (response.status === 400 || response.status === 409) {
    return "Refus du serveur. Aucun second dossier.";
  }
  return lostMessage;
}
