/**
 * Gate de séquence temps réel — Lot 3 perf espace entreprise.
 *
 * Le dashboard n'applique un événement Socket.IO que si son `event_seq` (curseur
 * monotone émis par le backend, voir backend/services/realtime/event_sequence.py)
 * est strictement supérieur :
 *   1. au `snapshot_cursor` du dernier bootstrap chargé (`GET /companies/me/dashboard/bootstrap`) ;
 *   2. au dernier `event_seq` déjà appliqué pour cette entreprise.
 *
 * `updated_at` seul n'est PAS un critère d'ordre valable (horloge client/serveur,
 * granularité seconde, retries) — c'est tout l'objet de ce garde-fou.
 *
 * Événements sans `event_seq` (legacy, ou payload chauffeur sans `company_id`) :
 * toujours acceptés (comportement identique à avant le Lot 3).
 */

const stateByCompany = new Map();

function getState(companyId) {
  const key = String(companyId);
  let state = stateByCompany.get(key);
  if (!state) {
    state = { snapshotCursor: 0, lastAppliedSeq: 0 };
    stateByCompany.set(key, state);
  }
  return state;
}

/** À appeler après chaque bootstrap réussi — nouveau curseur de référence. */
export function setSnapshotCursor(companyId, cursor) {
  if (companyId == null) return;
  const value = Number(cursor);
  if (!Number.isFinite(value) || value < 0) return;
  const state = getState(companyId);
  state.snapshotCursor = value;
  // Un nouveau bootstrap rattrape toujours l'état : jamais de régression du curseur appliqué.
  if (value > state.lastAppliedSeq) {
    state.lastAppliedSeq = value;
  }
}

/**
 * Décide si un événement temps réel doit être appliqué.
 *
 * @returns {{ accept: boolean, gapDetected: boolean }}
 *   - `accept`: true si l'événement doit être traité (ou s'il n'a pas de `event_seq`).
 *   - `gapDetected`: true si l'on saute au moins un `event_seq` (indice d'événements
 *     manqués — déclenche normalement un resync coalescé côté appelant).
 */
export function evaluateRealtimeSequence(companyId, eventSeq) {
  const seq = Number(eventSeq);
  if (!Number.isFinite(seq) || seq <= 0) {
    return { accept: true, gapDetected: false };
  }
  if (companyId == null) {
    return { accept: true, gapDetected: false };
  }
  const state = getState(companyId);
  const floor = Math.max(state.snapshotCursor, state.lastAppliedSeq);
  if (seq <= floor) {
    return { accept: false, gapDetected: false };
  }
  const gapDetected = state.lastAppliedSeq > 0 && seq > state.lastAppliedSeq + 1;
  state.lastAppliedSeq = seq;
  return { accept: true, gapDetected };
}

export function getLastAppliedSeq(companyId) {
  return getState(companyId).lastAppliedSeq;
}

export function getSnapshotCursor(companyId) {
  return getState(companyId).snapshotCursor;
}

/** Réinitialise l'état (déconnexion / changement d'entreprise). */
export function resetCompanySequenceState(companyId) {
  if (companyId == null) return;
  stateByCompany.delete(String(companyId));
}
