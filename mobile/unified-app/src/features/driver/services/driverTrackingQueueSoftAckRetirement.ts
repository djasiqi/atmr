/**
 * JZ-R1-DRAIN-BUDGET-FIX-23 — retraite bornée des heads soft-ACK sticky.
 *
 * Bornes : DOC_EXISTING (`docs/ops/gps-hol-ack-retirement-rca-closed-2026-08-25.md`)
 * — réimplémentées sur la lignée actuelle (aucun commit T2b retrouvé dans Git).
 *
 * Constantes figées (pas de EXPO_PUBLIC_) pour ne pas interagir avec
 * l'inlining babel / le cache Jest des budgets drain existants.
 *
 * Ne touche pas au budget minute / batch. Ne marque jamais `persisted`.
 */

/** DOC_EXISTING : MAX_AWAITING_DURABLE_ACK_ATTEMPTS (défaut 20). */
export const MAX_AWAITING_DURABLE_ACK_ATTEMPTS = 20;

/** DOC_EXISTING : MAX_HEAD_HOL_BLOCK_MS (défaut 15 min) — nom local sans HOL. */
export const MAX_HEAD_SOFT_ACK_BLOCK_MS = 15 * 60 * 1000;

export type SoftAckRetirementReason =
  | "ingested_non_persisted_expired"
  | "soft_ack_retry_exhausted";

const SOFT_ACK_LAST_ERRORS = new Set([
  "awaiting_durable_ack",
  "persisted_without_durability",
  "location_event_id_mismatch",
  "ack_event_id_mismatch",
  "partially_ingested_retry",
  "partially_ingested_current_missing",
  "partially_ingested_lists_missing",
  "partially_ingested_list_conflict",
]);

export type SoftAckRetirementItem = {
  queuedAt: number;
  retryCount: number;
  persistState?: string | null;
  lastError?: string | null;
  deliveryState?: string | null;
};

export function isSoftAckStickyItem(item: SoftAckRetirementItem): boolean {
  if (item.persistState === "ingested_non_persisted") return true;
  const err = item.lastError ?? "";
  if (SOFT_ACK_LAST_ERRORS.has(err)) return true;
  if (err.startsWith("unexpected_ack_")) return true;
  return false;
}

/**
 * Raison de retraite si la borne est atteinte ; sinon null (retry soft-ACK encore autorisé).
 * Ordre : tentatives d'abord (preuve d'effort), puis âge de blocage.
 */
export function softAckRetirementReason(
  item: SoftAckRetirementItem,
  nowMs: number,
  absoluteMaxAgeMs: number
): SoftAckRetirementReason | null {
  if (!isSoftAckStickyItem(item)) return null;
  if (item.retryCount >= MAX_AWAITING_DURABLE_ACK_ATTEMPTS) {
    return "soft_ack_retry_exhausted";
  }
  const ageMs = Math.max(0, nowMs - item.queuedAt);
  if (ageMs > absoluteMaxAgeMs) {
    return "ingested_non_persisted_expired";
  }
  if (ageMs >= MAX_HEAD_SOFT_ACK_BLOCK_MS) {
    return "ingested_non_persisted_expired";
  }
  return null;
}
