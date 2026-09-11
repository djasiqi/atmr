/**
 * Mesures temporaires — validation contrôle facturation (optimistic UX).
 * Activer les logs : localStorage.atmrBillingValidatePerf = '1'
 */

const PREFIX = 'billing_validate';

function nowMs() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function canLog() {
  if (typeof window === 'undefined') return false;
  try {
    return (
      process.env.NODE_ENV === 'development'
      || window.localStorage?.getItem('atmrBillingValidatePerf') === '1'
    );
  } catch {
    return process.env.NODE_ENV === 'development';
  }
}

function safeMark(name) {
  if (typeof performance === 'undefined' || !performance.mark) return;
  try {
    performance.mark(name);
  } catch {
    // ignore duplicate or invalid marks
  }
}

function debugLog(payload) {
  if (!canLog()) return;
  // Journal de perf temporaire, uniquement si le flag est actif
  console.info('[billing-control-validate]', payload); // eslint-disable-line no-console
}

export function markBillingValidateClick(bookingId) {
  safeMark(`${PREFIX}_click_${bookingId}`);
  const t = nowMs();
  debugLog({ event: 'billing_validate_click', bookingId, t });
  return t;
}

export function markBillingValidateUiCommit(bookingId, clickAt) {
  safeMark(`${PREFIX}_ui_commit_${bookingId}`);
  const t = nowMs();
  debugLog({
    event: 'billing_validate_ui_commit',
    bookingId,
    click_to_visual_feedback_ms: clickAt == null ? null : Math.round(t - clickAt),
  });
  return t;
}

export function markBillingValidateRequestStart(bookingId) {
  safeMark(`${PREFIX}_request_start_${bookingId}`);
  const t = nowMs();
  debugLog({ event: 'billing_validate_request_start', bookingId, t });
  return t;
}

export function markBillingValidateRequestEnd(bookingId, { clickAt, requestStartAt, ok } = {}) {
  safeMark(`${PREFIX}_request_end_${bookingId}`);
  const t = nowMs();
  debugLog({
    event: 'billing_validate_request_end',
    bookingId,
    ok: Boolean(ok),
    request_duration_ms: requestStartAt == null ? null : Math.round(t - requestStartAt),
    click_to_final_state_ms: clickAt == null ? null : Math.round(t - clickAt),
    number_of_network_requests: 1,
  });
  return t;
}

const PAYER_PREFIX = 'billing_payer';

export function markBillingPayerChange(bookingId) {
  safeMark(`${PAYER_PREFIX}_change_click_${bookingId}`);
  const t = nowMs();
  debugLog({ event: 'billing_payer_change_click', bookingId, t });
  return t;
}

export function markBillingPayerUiCommit(bookingId, clickAt) {
  safeMark(`${PAYER_PREFIX}_ui_commit_${bookingId}`);
  const t = nowMs();
  debugLog({
    event: 'billing_payer_ui_commit',
    bookingId,
    change_to_visual_feedback_ms: clickAt == null ? null : Math.round(t - clickAt),
  });
  return t;
}

export function markBillingPayerRequestStart(bookingId) {
  safeMark(`${PAYER_PREFIX}_request_start_${bookingId}`);
  const t = nowMs();
  debugLog({ event: 'billing_payer_request_start', bookingId, t });
  return t;
}

export function markBillingPayerRequestEnd(bookingId, { clickAt, requestStartAt, ok } = {}) {
  safeMark(`${PAYER_PREFIX}_request_end_${bookingId}`);
  const t = nowMs();
  debugLog({
    event: 'billing_payer_request_end',
    bookingId,
    ok: Boolean(ok),
    request_duration_ms: requestStartAt == null ? null : Math.round(t - requestStartAt),
    change_to_final_state_ms: clickAt == null ? null : Math.round(t - clickAt),
    network_requests_count: 1,
  });
  return t;
}
