/**
 * Gate de séquence temps réel — transactionnel (inspect → apply → commit).
 *
 * - inspectSequence : décide sans avancer lastAppliedSeq
 * - commitAppliedSequence : avance le curseur après effet réussi
 * - resyncRequired : trou détecté ; soldé uniquement via clearResyncAfterBootstrapSuccess
 */

const stateByCompany = new Map();

function getState(companyId) {
  const key = String(companyId);
  let state = stateByCompany.get(key);
  if (!state) {
    state = {
      snapshotCursor: null,
      lastAppliedSeq: 0,
      subscribedCursor: null,
      resyncRequired: false,
      realtimeDegraded: false,
      pendingBuffer: new Map(),
    };
    stateByCompany.set(key, state);
  }
  return state;
}

/**
 * @param {number|string|null} companyId
 * @param {number|null|undefined} cursor — null = Redis dégradé
 * @param {{ degraded?: boolean }} [opts]
 */
export function setSnapshotCursor(companyId, cursor, opts = {}) {
  if (companyId == null) return;
  const state = getState(companyId);
  if (cursor == null || !Number.isFinite(Number(cursor))) {
    state.snapshotCursor = null;
    state.realtimeDegraded = true;
    return;
  }
  const value = Number(cursor);
  if (value < 0) return;
  state.snapshotCursor = value;
  state.realtimeDegraded = Boolean(opts.degraded);
  if (value > state.lastAppliedSeq) {
    state.lastAppliedSeq = value;
  }
}

export function setSubscribedCursor(companyId, cursor) {
  if (companyId == null) return;
  const state = getState(companyId);
  if (cursor == null || !Number.isFinite(Number(cursor))) {
    state.subscribedCursor = null;
    state.realtimeDegraded = true;
    return;
  }
  state.subscribedCursor = Number(cursor);
}

/**
 * Inspecte sans commit. Rejette si company_id manquant ou event_seq invalide (≤0).
 * @returns {{ accept: boolean, gapDetected: boolean, reason?: string }}
 */
export function inspectSequence(companyId, eventSeq, payloadCompanyId = null) {
  if (companyId == null) {
    return { accept: false, gapDetected: false, reason: 'missing_company' };
  }
  if (payloadCompanyId != null && Number(payloadCompanyId) !== Number(companyId)) {
    return { accept: false, gapDetected: false, reason: 'company_mismatch' };
  }
  const seq = Number(eventSeq);
  if (!Number.isFinite(seq) || seq <= 0) {
    return { accept: false, gapDetected: false, reason: 'invalid_event_seq' };
  }
  const state = getState(companyId);
  if (state.realtimeDegraded || state.snapshotCursor == null) {
    return { accept: false, gapDetected: false, reason: 'realtime_degraded' };
  }
  const floor = Math.max(state.snapshotCursor || 0, state.lastAppliedSeq || 0);
  if (seq <= floor) {
    return { accept: false, gapDetected: false, reason: 'stale' };
  }
  const gapDetected = state.lastAppliedSeq > 0 && seq > state.lastAppliedSeq + 1;
  if (gapDetected) {
    state.resyncRequired = true;
  }
  return { accept: true, gapDetected };
}

/** @deprecated — préférer inspectSequence + commitAppliedSequence */
export function evaluateRealtimeSequence(companyId, eventSeq) {
  const result = inspectSequence(companyId, eventSeq);
  if (result.accept && !result.gapDetected) {
    commitAppliedSequence(companyId, eventSeq);
  } else if (result.accept && result.gapDetected) {
    // Ne pas committer sur un trou — resync d'abord
  }
  return { accept: result.accept, gapDetected: result.gapDetected };
}

export function commitAppliedSequence(companyId, eventSeq) {
  if (companyId == null) return;
  const seq = Number(eventSeq);
  if (!Number.isFinite(seq) || seq <= 0) return;
  const state = getState(companyId);
  if (seq > state.lastAppliedSeq) {
    state.lastAppliedSeq = seq;
  }
}

export function bufferRealtimeEvent(companyId, eventSeq, payload) {
  if (companyId == null) return;
  const seq = Number(eventSeq);
  if (!Number.isFinite(seq) || seq <= 0) return;
  getState(companyId).pendingBuffer.set(seq, payload);
}

export function drainContiguousBuffer(companyId) {
  const state = getState(companyId);
  const floor = Math.max(state.snapshotCursor || 0, state.lastAppliedSeq || 0);
  const sub = state.subscribedCursor;
  if (sub == null || state.snapshotCursor == null) {
    return { complete: false, events: [], resyncRequired: true };
  }
  const events = [];
  for (let s = floor + 1; s <= sub; s += 1) {
    if (!state.pendingBuffer.has(s)) {
      return { complete: false, events, resyncRequired: true };
    }
    events.push({ seq: s, payload: state.pendingBuffer.get(s) });
  }
  events.forEach(({ seq }) => state.pendingBuffer.delete(seq));
  return { complete: true, events, resyncRequired: false };
}

export function markResyncRequired(companyId) {
  if (companyId == null) return;
  getState(companyId).resyncRequired = true;
}

export function clearResyncAfterBootstrapSuccess(companyId) {
  if (companyId == null) return;
  const state = getState(companyId);
  state.resyncRequired = false;
  state.pendingBuffer.clear();
}

export function isResyncRequired(companyId) {
  return Boolean(getState(companyId).resyncRequired);
}

export function isRealtimeDegraded(companyId) {
  return Boolean(getState(companyId).realtimeDegraded || getState(companyId).snapshotCursor == null);
}

export function getLastAppliedSeq(companyId) {
  return getState(companyId).lastAppliedSeq;
}

export function getSnapshotCursor(companyId) {
  return getState(companyId).snapshotCursor;
}

export function getSubscribedCursor(companyId) {
  return getState(companyId).subscribedCursor;
}

export function resetCompanySequenceState(companyId) {
  if (companyId == null) return;
  stateByCompany.delete(String(companyId));
}
