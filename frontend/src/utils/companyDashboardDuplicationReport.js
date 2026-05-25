/**
 * Rapport duplication API dashboard (dispatch_mode, alerts, …).
 */

import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';

const calls = [];
const mountCounters = {};

function parseCallerFromStack(stack) {
  if (!stack || typeof stack !== 'string') return 'unknown';
  const lines = stack.split('\n').slice(1, 8);
  for (const line of lines) {
    if (line.includes('useDispatchMode')) return 'useDispatchMode';
    if (line.includes('CompanyNotificationBell')) return 'CompanyNotificationBell';
    if (line.includes('CompanyDashboard')) return 'CompanyDashboard';
    if (line.includes('DispatchModeSelector')) return 'DispatchModeSelector';
  }
  return lines[0]?.trim() || 'unknown';
}

function getDashStartTime() {
  if (typeof performance === 'undefined' || !performance.getEntriesByName) return 0;
  const marks = performance.getEntriesByName('dashboard_start', 'mark');
  return marks.length ? marks[marks.length - 1].startTime : 0;
}

/**
 * @param {{ key: string, url?: string, componentId?: string, callerStack?: string }} opts
 */
export function recordDashboardApiCall(opts) {
  if (!isCompanyDashboardPerfEnabled()) return;
  const { key, url, componentId, callerStack } = opts || {};
  if (!key) return;

  const compId = componentId || 'default';
  mountCounters[compId] = (mountCounters[compId] || 0) + 1;
  const dashStart = getDashStartTime();
  const now = typeof performance !== 'undefined' ? performance.now() : Date.now();

  calls.push({
    key,
    url: url || null,
    callIndex: calls.filter((c) => c.key === key).length + 1,
    atMs: dashStart ? Math.round(now - dashStart) : Math.round(now),
    caller: parseCallerFromStack(callerStack),
    effectId: `${compId}#${mountCounters[compId]}`,
    recordedAt: Date.now(),
  });
}

function inferCauseForKey(key, keyCalls) {
  if (keyCalls.length <= 1) return 'single';
  const gapMs =
    keyCalls.length >= 2
      ? Math.abs((keyCalls[1].atMs || 0) - (keyCalls[0].atMs || 0))
      : 0;
  if (gapMs < 50) return 'strict_mode_candidate';
  return 'true_duplicate_candidate';
}

export function buildDuplicationReport(devProdDiff = null) {
  const perKey = {};
  for (const call of calls) {
    if (!perKey[call.key]) {
      perKey[call.key] = { count: 0, callers: new Set(), calls: [] };
    }
    perKey[call.key].count += 1;
    perKey[call.key].callers.add(call.caller);
    perKey[call.key].calls.push(call);
  }

  const perKeySummary = {};
  Object.entries(perKey).forEach(([key, data]) => {
    const cause = inferCauseForKey(key, data.calls);
    let resolvedCause = cause;
    if (devProdDiff && devProdDiff[key]) {
      const { devCount, prodCount } = devProdDiff[key];
      if (devCount >= 2 && prodCount <= 1) resolvedCause = 'strict_mode';
      else if (devCount >= 2 && prodCount >= 2) resolvedCause = 'true_duplicate';
      else if (devCount <= 1) resolvedCause = 'single';
      else resolvedCause = 'unknown';
    } else if (cause === 'strict_mode_candidate') {
      resolvedCause = 'unknown';
    } else if (cause === 'true_duplicate_candidate') {
      resolvedCause = 'unknown';
    }

    perKeySummary[key] = {
      count: data.count,
      callers: [...data.callers],
      cause: resolvedCause,
    };
  });

  return {
    title: 'Rapport Duplication — appels API dashboard',
    generatedAt: new Date().toISOString(),
    calls: calls.map((c) => ({
      ...c,
      cause: perKeySummary[c.key]?.cause,
    })),
    perKey: perKeySummary,
    note: 'Comparer dev (StrictMode) vs prod-build pour classer strict_mode vs true_duplicate',
  };
}

export function publishDuplicationReport(devProdDiff = null) {
  const report = buildDuplicationReport(devProdDiff);
  if (typeof window !== 'undefined') {
    window.__companyDashboardDuplicationReport = report;
  }
  return report;
}

export function resetDuplicationReport() {
  calls.length = 0;
  Object.keys(mountCounters).forEach((k) => delete mountCounters[k]);
}
