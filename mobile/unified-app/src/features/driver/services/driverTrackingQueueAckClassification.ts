/**
 * JZ-R1-STALE-SOFT-ACK-FIX-31 — classification minimale des soft-ACK HTTP.
 *
 * Ne terminalise que `too_old_for_mode` (irréversible).
 * Les autres ingested_non_persisted restent retryables (branche existante).
 */

export const BACKEND_TOO_OLD_FOR_MODE_REASON = "backend_too_old_for_mode";

export type SoftAckQueueDecision =
  | { kind: "terminal_backend_too_old" }
  | { kind: "continue" };

export type SoftAckClassifyInput = {
  ack_status: string;
  accept_reason?: string | null;
};

/**
 * Décision queue pour un ACK déjà parsé.
 * Branche ciblée : ingested_non_persisted + too_old_for_mode uniquement.
 */
export function classifySoftAckForQueue(
  ack: SoftAckClassifyInput
): SoftAckQueueDecision {
  if (
    ack.ack_status === "ingested_non_persisted" &&
    ack.accept_reason === "too_old_for_mode"
  ) {
    return { kind: "terminal_backend_too_old" };
  }
  return { kind: "continue" };
}
