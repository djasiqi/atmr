export type SettlementMode = "tap_to_pay" | "cash" | "invoice";

export type SettlementKind =
  | "already_paid"
  | "collection_authorized"
  | "invoice_authorized"
  | "no_collection"
  | "collection_forbidden"
  | "administrative_review";

/** Contrat reçu du backend ou d'une fixture. Le téléphone ne le recalcule pas. */
export type DriverSettlementContract = {
  kind: SettlementKind;
  reason: string | null;
  modes: SettlementMode[];
  amount_chf: string | null;
  currency: "CHF";
  payment_scope: "booking" | "group" | null;
  simulation: boolean;
  payment_executable: false;
  persists_payment: false;
  provider: "none" | "simulation";
  stale: boolean;
  authorization_durable: false;
  recheck_required: true;
  attestation_separate: true;
  completion_independent: true;
  driver_may_edit_amount: false;
  driver_may_edit_payer: false;
  writes_completed_payment: false;
  writes_cash_receipt: false;
  writes_invoice: false;
  writes_financial_signature: false;
  invoice_means_paid: false;
  source: "backend" | "simulation";
  company_id: number | null;
  debtor_label: string | null;
  document_approved: boolean;
  document_version: string | null;
  signature_required: boolean;
  attestation_recorded: boolean;
  payment_received: boolean;
  preview_screen: string | null;
};

export type SettlementSurface =
  | "hidden"
  | "loading"
  | "error"
  | "discreet"
  | "review"
  | "simulation";

const QUIET_KINDS = new Set<SettlementKind>([
  "already_paid",
  "no_collection",
  "collection_forbidden",
]);

export function settlementSurface(
  contract: DriverSettlementContract | null,
  options: { loading?: boolean; error?: boolean; offline?: boolean }
): SettlementSurface {
  if (options.loading) return "loading";
  if (options.error || options.offline) return "error";
  if (!contract) return "hidden";
  if (contract.stale || contract.authorization_durable) return "review";
  if (contract.payment_executable !== false || contract.persists_payment !== false) {
    return "review";
  }
  const simulated =
    contract.simulation &&
    contract.source === "simulation" &&
    contract.provider === "simulation" &&
    contract.modes.length > 0 &&
    (contract.kind === "collection_authorized" || contract.kind === "invoice_authorized");
  if (simulated) return "simulation";
  if (contract.kind === "administrative_review") return "review";
  if (QUIET_KINDS.has(contract.kind)) return "discreet";
  return "review";
}

export function discreetMessage(contract: DriverSettlementContract): string {
  if (contract.kind === "already_paid" || contract.payment_received) {
    return "Ce transport est déjà payé. Aucun règlement sur place.";
  }
  if (contract.reason === "institution_payer" || contract.reason === "institution_account") {
    return "L'institution prend en charge ce transport. Aucun règlement sur place.";
  }
  if (contract.reason === "established_account" || contract.reason === "third_party_convention") {
    return "Un compte ou une convention existe déjà. Aucun règlement sur place.";
  }
  if (contract.reason === "guest_flow_unchanged") {
    return "Le parcours invité reste inchangé. Aucun règlement sur place.";
  }
  if (contract.reason === "payer_still_mutable") {
    return "Le payeur peut encore changer. Aucun règlement sur place.";
  }
  if (
    contract.reason === "company_not_owner" ||
    contract.reason === "driver_unassigned" ||
    contract.reason === "other_company"
  ) {
    return "Cette mission n'est pas accessible pour un règlement.";
  }
  if (contract.reason === "collection_disabled") {
    return "Aucun moyen de règlement n'est activé.";
  }
  return "Aucun règlement sur place n'est demandé pour cette mission.";
}

export function visibleModes(contract: DriverSettlementContract): SettlementMode[] {
  if (settlementSurface(contract, {}) !== "simulation") return [];
  return contract.modes;
}

export function contractAllowsMock(contract: DriverSettlementContract): boolean {
  return (
    settlementSurface(contract, {}) === "simulation" &&
    contract.writes_completed_payment === false &&
    contract.writes_cash_receipt === false &&
    contract.writes_invoice === false &&
    contract.writes_financial_signature === false
  );
}

const KINDS = new Set<SettlementKind>([
  "already_paid",
  "collection_authorized",
  "invoice_authorized",
  "no_collection",
  "collection_forbidden",
  "administrative_review",
]);

const MODES = new Set<SettlementMode>(["tap_to_pay", "cash", "invoice"]);

function asRecord(payload: unknown): Record<string, unknown> | null {
  if (!payload || typeof payload !== "object") return null;
  const record = payload as Record<string, unknown>;
  const inner = record.data;
  if (inner && typeof inner === "object" && "kind" in (inner as object)) {
    return inner as Record<string, unknown>;
  }
  if ("kind" in record) return record;
  return null;
}

/**
 * Le téléphone affiche le contrat reçu. Hors prévisualisation, aucun mode
 * n'est conservé, même si une réponse prétend autoriser un encaissement.
 */
export function presentServerDecision(
  payload: unknown,
  options: { preview: boolean }
): DriverSettlementContract | null {
  const raw = asRecord(payload);
  if (!raw) return null;
  const kind = KINDS.has(raw.kind as SettlementKind)
    ? (raw.kind as SettlementKind)
    : "administrative_review";
  const source = raw.source === "simulation" ? "simulation" : "backend";
  const preview = options.preview && source === "simulation" && raw.simulation === true;
  const modes = preview
    ? ((Array.isArray(raw.modes) ? raw.modes : []).filter((mode) =>
        MODES.has(mode as SettlementMode)
      ) as SettlementMode[])
    : [];
  const executableKind = kind === "collection_authorized" || kind === "invoice_authorized";
  const shownKind = preview || !executableKind ? kind : "administrative_review";
  const scope =
    raw.payment_scope === "group" ? "group" : raw.payment_scope === "booking" ? "booking" : null;
  return {
    kind: shownKind,
    reason: typeof raw.reason === "string" ? raw.reason : null,
    modes,
    amount_chf: typeof raw.amount_chf === "string" ? raw.amount_chf : null,
    currency: "CHF",
    payment_scope: scope,
    simulation: preview,
    payment_executable: false,
    persists_payment: false,
    provider: preview ? "simulation" : "none",
    stale: raw.stale === true,
    authorization_durable: false,
    recheck_required: true,
    attestation_separate: true,
    completion_independent: true,
    driver_may_edit_amount: false,
    driver_may_edit_payer: false,
    writes_completed_payment: false,
    writes_cash_receipt: false,
    writes_invoice: false,
    writes_financial_signature: false,
    invoice_means_paid: false,
    source: preview ? "simulation" : "backend",
    company_id: typeof raw.company_id === "number" ? raw.company_id : null,
    debtor_label: typeof raw.debtor_label === "string" ? raw.debtor_label : null,
    document_approved: raw.document_approved === true,
    document_version: typeof raw.document_version === "string" ? raw.document_version : null,
    signature_required: raw.signature_required === true,
    attestation_recorded: raw.attestation_recorded === true,
    payment_received: raw.payment_received === true || kind === "already_paid",
    preview_screen: preview && typeof raw.preview_screen === "string" ? raw.preview_screen : null,
  };
}

