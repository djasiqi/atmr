/**
 * Le Dashboard n'ouvre le règlement que si la lecture serveur l'autorise.
 * Le téléphone ne déduit pas le payeur et n'active pas Tap to Pay.
 */

function contractOf(payload: unknown): Record<string, unknown> | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const inner = record.data;
  if (inner && typeof inner === "object" && "kind" in (inner as object)) {
    return inner as Record<string, unknown>;
  }
  if ("kind" in record) return record;
  return null;
}

const REAL_WRITE = [
  "payment_executable",
  "persists_payment",
  "authorization_durable",
  "writes_completed_payment",
  "writes_cash_receipt",
  "writes_invoice",
  "writes_financial_signature",
] as const;

export type SettlementModes = { cash: boolean; invoice: boolean };

/**
 * Espèces et facture patient simulées. Tap to Pay n'est jamais une modalité active.
 * Une écriture réelle, une lecture périmée ou un refus restent fermés.
 */
/**
 * Feuille du registre V1 local. Ce n'est ni une simulation, ni un paiement bancaire.
 * Le serveur a déjà passé les gardes métier.
 */
export function v1LocalCashAuthorized(payload: unknown): boolean {
  const source = contractOf(payload);
  if (!source) return false;
  if (REAL_WRITE.some((key) => source[key] === true)) return false;
  const sheet = source.cash_sheet;
  if (!sheet || typeof sheet !== "object") return false;
  const row = sheet as {
    allowed?: unknown;
    register?: unknown;
    simulation?: unknown;
    real_collection?: unknown;
  };
  return (
    row.allowed === true &&
    row.register === "cash_v1" &&
    row.simulation !== true &&
    row.real_collection !== true
  );
}

export function dashboardSettlementModes(payload: unknown): SettlementModes {
  const source = contractOf(payload);
  const closed = { cash: false, invoice: false };
  if (!source) return closed;
  if (REAL_WRITE.some((key) => source[key] === true)) return closed;
  if (v1LocalCashAuthorized(payload)) return { cash: true, invoice: false };
  if (source.simulation !== true || source.source !== "simulation") return closed;
  if (source.stale === true) return closed;
  const modes = Array.isArray(source.modes) ? source.modes : [];
  const cash = source.kind === "collection_authorized" && modes.includes("cash");
  const invoice =
    modes.includes("invoice") &&
    (source.kind === "collection_authorized" || source.kind === "invoice_authorized");
  return { cash, invoice };
}

export function dashboardSettlementAuthorized(payload: unknown): boolean {
  const modes = dashboardSettlementModes(payload);
  return modes.cash || modes.invoice;
}

/**
 * La situation financière n'est pas décidée.
 * Une rubrique absente ou un contrat sans politique ne signifie pas « aucun règlement ».
 */
const UNDETERMINED_REASONS = new Set([
  "settlement_policy_missing",
  "financial_policy_unavailable",
  "collection_disabled",
  "no_authorized_mode",
]);

export function dashboardVerificationRequired(payload: unknown): boolean {
  const source = contractOf(payload);
  if (!source) return false;
  if (v1LocalCashAuthorized(payload)) return false;
  if (source.verification_required === true || source.kind === "financial_situation_unconfirmed") {
    return true;
  }
  if (source.simulation === true || source.kind === "collection_authorized") return false;
  if (source.kind === "invoice_authorized") return false;
  if (source.kind === "collection_forbidden" || source.kind === "already_paid") return false;
  if (source.kind === "no_collection") return false;
  return source.kind === "administrative_review" && UNDETERMINED_REASONS.has(String(source.reason));
}

/**
 * Sixième étape si le serveur autorise un règlement chauffeur : espèces ou facture
 * encore à confirmer. Un mandat applicable, une institution ou une situation
 * indéterminée ne l'ouvrent pas.
 */
export function dashboardDestinationStepAuthorized(payload: unknown): boolean {
  if (dashboardVerificationRequired(payload)) return false;
  const modes = dashboardSettlementModes(payload);
  if (modes.cash) return true;
  return modes.invoice && invoiceConfirmationRequired(payload);
}

/**
 * Un mandat de démonstration applicable prépare la facture sans demander « Facture ».
 * L'étape d'encaissement reste réservée aux espèces.
 */
export function invoiceConfirmationRequired(payload: unknown): boolean {
  const source = contractOf(payload);
  const follow = source?.billing_followup;
  if (!follow || typeof follow !== "object") return true;
  return (follow as { invoice_confirmation_required?: unknown }).invoice_confirmation_required !== false;
}

/**
 * Espèces exceptionnelles depuis les Détails. Le tableau de bord ne gagne pas d'étape.
 * Le téléphone n'autorise rien que le serveur n'a pas marqué.
 */
export function exceptionalCashOffer(payload: unknown): string | null {
  const source = contractOf(payload);
  const offer = source?.exceptional_cash;
  if (!offer || typeof offer !== "object") return null;
  const record = offer as { allowed?: unknown; label?: unknown };
  if (record.allowed !== true) return null;
  if (dashboardSettlementModes(payload).cash) return null;
  return typeof record.label === "string" && record.label.trim()
    ? record.label.trim()
    : "Encaisser en espèces";
}

export function billingFollowupLabel(payload: unknown): string | null {
  const source = contractOf(payload);
  const follow = source?.billing_followup;
  if (!follow || typeof follow !== "object") return null;
  if (invoiceConfirmationRequired(payload)) return null;
  const record = follow as { label?: unknown; dossier_state?: unknown };
  const label = record.label;
  if (typeof label !== "string" || !label.trim()) return null;
  const text = label.trim();
  const state = record.dossier_state;
  if (text.includes("Dossier préparé") && state !== "prepared") {
    return "Mandat signé. Facturation prévue, dossier non préparé. À facturer.";
  }
  return text;
}

const SETTLED_CASH_STATES = new Set(["received", "returned", "administrative", "uncertain"]);

/** Une réception, un partiel, un refus ou une réponse perdue. « Préparé » ne compte pas. */
export function cashCollectionSettled(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  const state = (payload as { state?: unknown }).state;
  return typeof state === "string" && SETTLED_CASH_STATES.has(state);
}

/** Un dossier patient déjà ouvert. Un 404 n'est pas une préparation. */
export function invoicePreparationStarted(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  const row = payload as { id?: unknown };
  return row.id != null;
}

export type DashboardPrimaryKind = "transition" | "destination" | "reopen";

/**
 * Tant que le règlement autorisé n'a pas d'issue, l'action principale est l'arrêt.
 * « Terminer » ne revient qu'ensuite, ou lorsqu'aucun règlement chauffeur n'est dû.
 */
export function dashboardPrimaryKind(input: {
  nextTransition: string | null;
  destinationAuthorized: boolean;
  destinationConfirmed: boolean;
  settlementSettled: boolean;
}): DashboardPrimaryKind {
  const finishing = (input.nextTransition ?? "").trim().toUpperCase() === "COMPLETED";
  if (finishing && input.destinationAuthorized && !input.settlementSettled) {
    return input.destinationConfirmed ? "reopen" : "destination";
  }
  return "transition";
}

function moneyLabel(raw: unknown): string | null {
  if (typeof raw === "string" && raw.trim()) return `${raw.trim()} CHF`;
  if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) {
    return `${raw.toFixed(2)} CHF`;
  }
  return null;
}

/** Montant exigible de la lecture. Le téléphone n'invente pas un prix. */
export function dashboardAmountLabel(payload: unknown): string {
  return confirmedAmountLabel(payload) ?? "Montant de la réservation";
}

/** Espèces simulées seulement. La facture seule n'ouvre pas le parcours espèces. */
export function dashboardCashAuthorized(payload: unknown): boolean {
  return dashboardSettlementModes(payload).cash;
}

const CLOSED_READ = new Set(["Rubrique indisponible", "Mission indisponible"]);

/**
 * Un 404 de rubrique fermée ou de mission inaccessible n'est pas une panne.
 * Il n'autorise aucune opération.
 */
/**
 * Le texte « conservé » n'est affiché qu'après une confirmation serveur.
 * Un refus ou une file locale laisse la mission terminable, sans paiement.
 */
export function settlementImpossibleNotice(result: "sent" | "queued" | "rejected"): string {
  if (result === "sent") {
    return "Signalement conservé. Aucune déclaration financière. La mission peut être terminée.";
  }
  if (result === "rejected") {
    return "Signalement non enregistré. Le serveur a refusé cet envoi. Il ne sera pas renvoyé automatiquement. La mission peut être terminée.";
  }
  return "Signalement non enregistré. L'envoi pourra être repris. Aucune déclaration financière. La mission peut être terminée.";
}

export function settlementRubricClosed(error: unknown): boolean {
  if (!error || typeof error !== "object" || !("response" in error)) return false;
  const data = (error as { response?: { data?: { error?: unknown } } }).response?.data;
  return Boolean(data && data.error === "Rubrique indisponible");
}

function confirmedAmountLabel(payload: unknown): string | null {
  const source = contractOf(payload);
  const direct = moneyLabel(source?.amount_chf);
  if (direct) return direct;
  const follow = source?.billing_followup;
  if (!follow || typeof follow !== "object") return null;
  return moneyLabel((follow as { due_chf?: unknown }).due_chf);
}

/**
 * La feuille d'espèces du registre V1 ne s'ouvre que si le serveur l'autorise.
 * Sans ce drapeau, une revue administrative reste fermée : ce n'est pas une panne.
 */
export function cashSheetClosedLine(payload: unknown): string | null {
  const source = contractOf(payload);
  if (!source) return null;
  const sheet = source.cash_sheet;
  if (sheet && typeof sheet === "object") {
    if ((sheet as { allowed?: unknown }).allowed === true) return null;
    const reason = (sheet as { reason?: unknown }).reason;
    if (typeof reason === "string" && reason.trim()) {
      const code = reason.trim();
      // Nouvelle collecte interdite ≠ historique illisible.
      if (code === "facture_emise" || code === "deja_paye") {
        return `Nouvelle collecte interdite (${code}).`;
      }
      return `Feuille espèces refusée (${code}).`;
    }
    return "Feuille espèces refusée.";
  }
  if (source.simulation === true) return null;
  const modes = Array.isArray(source.modes) ? source.modes : [];
  if (modes.includes("cash")) return null;
  if (source.kind === "administrative_review") {
    return "Feuille espèces fermée. La lecture n'autorise pas les espèces.";
  }
  return null;
}

export function verificationNotice(payload: unknown): string {
  const source = contractOf(payload);
  const billed = source?.billed_to_type;
  const payer =
    billed === "patient" ? "Patient" : billed === "clinic" || billed === "institution" ? "Institution" : null;
  const detail = [payer, confirmedAmountLabel(payload)].filter(Boolean).join(", ");
  const base = !detail
    ? "Vérification du règlement nécessaire"
    : `Vérification du règlement nécessaire. ${detail}. Aucune politique de règlement.`;
  const closed = cashSheetClosedLine(payload);
  return closed ? `${base} ${closed}` : base;
}

export function handoverReadClosed(error: unknown): boolean {
  const response = (error as { response?: { status?: number; data?: { error?: unknown } } } | null)
    ?.response;
  if (response?.status !== 404) return false;
  return typeof response.data?.error === "string" && CLOSED_READ.has(response.data.error);
}
