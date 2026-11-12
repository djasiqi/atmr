/**
 * Prototype d'encaissement chauffeur. Aucune route ne l'affiche.
 * L'activation produit reste coupée : aucun débit n'est lancé.
 */

export const END_OF_TRIP_COLLECTION_ENABLED = false;

export type TapState =
  | "ready"
  | "in_progress"
  | "confirmed"
  | "refused"
  | "uncertain"
  | "already_done";

const OPEN_BLOCK = new Set<TapState>(["in_progress", "uncertain"]);

export type TapView = {
  state: TapState;
  attemptId: string | null;
  mayStartAnotherDebit: false;
};

export function collectionScreenVisible(
  enabled: boolean,
  collectionRequired: boolean,
  ready: boolean
): boolean {
  return enabled && collectionRequired && ready;
}

export function openTapAttempt(
  current: TapState,
  attemptId: string | null,
  nextAttemptId: string
): TapView {
  if (current === "confirmed" || current === "already_done") {
    return { state: "already_done", attemptId, mayStartAnotherDebit: false };
  }
  if (OPEN_BLOCK.has(current)) {
    return { state: "uncertain", attemptId, mayStartAnotherDebit: false };
  }
  const next = nextAttemptId.trim();
  if (!next) {
    throw new Error("tentative_requise");
  }
  return { state: "in_progress", attemptId: next, mayStartAnotherDebit: false };
}

export function resolveTapSimulation(
  current: TapState,
  attemptId: string | null,
  result: "confirmed" | "refused" | "uncertain"
): TapView {
  if (current !== "in_progress") {
    return {
      state: current === "confirmed" ? "already_done" : current,
      attemptId,
      mayStartAnotherDebit: false,
    };
  }
  return { state: result, attemptId, mayStartAnotherDebit: false };
}

export type CashReceipt = "missing" | "declared" | "difference";

/** Un appui sur « Espèces » sans montant reçu ne vaut pas un paiement. */
export function cashReceiptDecision(
  expected: string,
  received: string | null,
  confirmed: boolean
): CashReceipt {
  if (!confirmed || received == null || received.trim() === "") {
    return "missing";
  }
  return received.trim() === expected.trim() ? "declared" : "difference";
}

export const INVOICE_SIGNATURE_DISCLAIMER =
  "Ce document est distinct de l'attestation de réalisation du trajet. " +
  "Il reprend uniquement le texte de facturation approuvé, sa version, " +
  "le montant et le débiteur désigné.";
