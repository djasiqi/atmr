/**
 * Métriques locales Socket.IO entreprise (debug / pré-rollout).
 * Activer via REACT_APP_SOCKET_METRICS_DEBUG=true ou localStorage company_socket_metrics_debug=1
 */

const metrics = {
  reconnectAttemptCount: 0,
  activeListenerCount: 0,
  activeListenersByEvent: {},
  connectToReadySamplesMs: [],
  connectToReadyAvgMs: 0,
  connectToReadyMinMs: null,
  connectToReadyMaxMs: null,
  connectStartedAtMs: null,
};

export function isCompanySocketMetricsDebugEnabled() {
  if (process.env.REACT_APP_SOCKET_METRICS_DEBUG === 'true') return true;
  if (typeof window === 'undefined') return false;
  try {
    return window.localStorage?.getItem('company_socket_metrics_debug') === '1';
  } catch {
    return false;
  }
}

function recomputeConnectToReadyStats() {
  const samples = metrics.connectToReadySamplesMs;
  if (samples.length === 0) {
    metrics.connectToReadyAvgMs = 0;
    metrics.connectToReadyMinMs = null;
    metrics.connectToReadyMaxMs = null;
    return;
  }
  const sum = samples.reduce((acc, v) => acc + v, 0);
  metrics.connectToReadyAvgMs = Math.round(sum / samples.length);
  metrics.connectToReadyMinMs = Math.min(...samples);
  metrics.connectToReadyMaxMs = Math.max(...samples);
}

export function recordReconnectAttempt() {
  metrics.reconnectAttemptCount += 1;
  if (isCompanySocketMetricsDebugEnabled()) {
    // eslint-disable-next-line no-console
    console.debug('[CompanySocketMetrics] reconnect_attempt', metrics.reconnectAttemptCount);
  }
}

export function recordConnectStarted() {
  metrics.connectStartedAtMs = Date.now();
}

export function recordConnectReady() {
  if (metrics.connectStartedAtMs == null) return;
  const durationMs = Date.now() - metrics.connectStartedAtMs;
  metrics.connectStartedAtMs = null;
  metrics.connectToReadySamplesMs.push(durationMs);
  if (metrics.connectToReadySamplesMs.length > 50) {
    metrics.connectToReadySamplesMs.shift();
  }
  recomputeConnectToReadyStats();
  if (isCompanySocketMetricsDebugEnabled()) {
    // eslint-disable-next-line no-console
    console.debug('[CompanySocketMetrics] connect_to_ready_ms', durationMs);
  }
}

export function setActiveListenerCount(total, byEvent = {}) {
  metrics.activeListenerCount = total;
  metrics.activeListenersByEvent = { ...byEvent };
}

export function getCompanySocketMetricsSnapshot() {
  return {
    reconnectAttemptCount: metrics.reconnectAttemptCount,
    activeListenerCount: metrics.activeListenerCount,
    activeListenersByEvent: { ...metrics.activeListenersByEvent },
    connectToReadyAvgMs: metrics.connectToReadyAvgMs,
    connectToReadyMinMs: metrics.connectToReadyMinMs,
    connectToReadyMaxMs: metrics.connectToReadyMaxMs,
  };
}

/** Dev: avertir si trop d'abonnés sur un même événement. */
export const MAX_DEV_LISTENERS_PER_EVENT = 12;

const listenerCountHistory = [];
const LISTENER_WATCHDOG_WINDOW_MS = 120_000;
const LISTENER_WATCHDOG_SLOPE_THRESHOLD = 6;

export function warnIfTooManyListeners(event, count) {
  if (process.env.NODE_ENV === 'production') return;
  if (count <= MAX_DEV_LISTENERS_PER_EVENT) return;
  // eslint-disable-next-line no-console
  console.warn(
    `[CompanySocket] ${count} listeners on "${event}" (seuil dev: ${MAX_DEV_LISTENERS_PER_EVENT})`
  );
}

export function trackListenerCountForWatchdog(total) {
  if (process.env.NODE_ENV === 'production') return;
  const now = Date.now();
  listenerCountHistory.push({ at: now, total });
  while (listenerCountHistory.length > 0 && now - listenerCountHistory[0].at > LISTENER_WATCHDOG_WINDOW_MS) {
    listenerCountHistory.shift();
  }
  if (listenerCountHistory.length < 2) return;
  const oldest = listenerCountHistory[0];
  const slope = total - oldest.total;
  if (slope >= LISTENER_WATCHDOG_SLOPE_THRESHOLD) {
    // eslint-disable-next-line no-console
    console.warn(
      `[CompanySocket] dérive listeners possible (+${slope} sur ${LISTENER_WATCHDOG_WINDOW_MS}ms)`
    );
  }
}
