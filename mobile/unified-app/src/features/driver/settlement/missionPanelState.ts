import {
  discreetMessage,
  presentServerDecision,
  settlementSurface,
  type DriverSettlementContract,
} from "./driverSettlementContract";
import {
  settlementClosed,
  settlementMomentOpen,
  settlementNeedsHalt,
} from "./settlementAccess";

export type MissionPanelState = {
  /** Panneau de simulation, uniquement après une ouverture volontaire. */
  visible: boolean;
  /** Confirmation d'arrêt. Ce n'est pas un bouton de paiement. */
  showHaltPrompt: boolean;
  /** Action facultative « Règlement du transport ». */
  showSettlementAction: boolean;
  /** Décision sans moyen actionnable. */
  notice: string | null;
  /** Consultation d'une mission déjà terminée. */
  consultation: boolean;
  contract: DriverSettlementContract | null;
  loading: boolean;
  offline: boolean;
  error: string | null;
};

const hidden: MissionPanelState = {
  visible: false,
  showHaltPrompt: false,
  showSettlementAction: false,
  notice: null,
  consultation: false,
  contract: null,
  loading: false,
  offline: false,
  error: null,
};

function claimedExecutable(payload: unknown): boolean {
  if (!payload || typeof payload !== "object") return false;
  const record = payload as Record<string, unknown>;
  const inner = record.data;
  const source =
    inner && typeof inner === "object" ? (inner as Record<string, unknown>) : record;
  return (
    source.authorization_durable === true ||
    source.payment_executable === true ||
    source.persists_payment === true
  );
}

/**
 * Le règlement ne s'ouvre pas parce que le statut est « Patient à bord ».
 * Le chauffeur confirme l'arrêt, puis ouvre l'action. Une mission terminée
 * se consulte sans rouvrir la course. Une décision réelle ne donne aucun moyen.
 */
export function missionPanelState(input: {
  allowed: boolean;
  operationalStatus: string | null;
  haltConfirmed: boolean;
  opened: boolean;
  offline: boolean;
  loading: boolean;
  failed: boolean;
  payload: unknown;
}): MissionPanelState {
  const consultation = settlementClosed(input.operationalStatus);
  if (!input.allowed || !settlementMomentOpen(input.operationalStatus)) return hidden;
  if (settlementNeedsHalt(input.operationalStatus) && !input.haltConfirmed) {
    return { ...hidden, showHaltPrompt: true };
  }
  if (!input.opened) {
    return { ...hidden, showSettlementAction: true, consultation };
  }
  if (input.offline) {
    return { ...hidden, offline: true, error: "offline", consultation };
  }
  if (input.failed) {
    return { ...hidden, error: "unavailable", consultation };
  }
  if (input.loading || input.payload == null) {
    return { ...hidden, loading: true, consultation };
  }
  const contract = presentServerDecision(input.payload, { preview: true });
  const unsafe = claimedExecutable(input.payload);
  const simulated =
    contract !== null &&
    contract.simulation &&
    contract.modes.length > 0 &&
    !contract.stale &&
    !unsafe;
  if (contract && simulated) {
    return { ...hidden, visible: true, contract, consultation };
  }
  return {
    ...hidden,
    notice: noticeFor(contract, unsafe),
    contract: contract ? { ...contract, modes: [] } : null,
    consultation,
  };
}

const QUIET_REASONS = new Set<string>([
  "payer_still_mutable",
  "collection_disabled",
  "company_not_owner",
  "driver_unassigned",
  "other_company",
  "institution_payer",
  "institution_account",
  "established_account",
  "third_party_convention",
  "guest_flow_unchanged",
  "settlement_policy_missing",
  "payment_confirmed",
]);

function noticeFor(contract: DriverSettlementContract | null, unsafe: boolean): string {
  if (!contract || contract.stale || unsafe) {
    return "Règlement à vérifier par l'entreprise.";
  }
  if (contract.reason && QUIET_REASONS.has(contract.reason)) {
    return discreetMessage(contract);
  }
  if (settlementSurface(contract, {}) === "review") {
    return "Règlement à vérifier par l'entreprise.";
  }
  return discreetMessage(contract);
}
