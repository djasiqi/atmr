/**
 * Instrumentation perf dashboard entreprise (dev / session MAP_DEBUG).
 * Baseline 5 min : window.__companyDashboardPerfBaseline.start() puis .report()
 */

function readPerfEnabled() {
  if (typeof window === 'undefined') {
    return process.env.NODE_ENV === 'development';
  }
  return (
    process.env.NODE_ENV === 'development' ||
    window.__MAP_DEBUG === true ||
    sessionStorage.getItem('MAP_DEBUG') === '1' ||
    sessionStorage.getItem('COMPANY_DASH_PERF') === '1'
  );
}

// Compatibilité avec l'ancienne constante tout en gardant l'activation dynamique.
const PERF_ENABLED = readPerfEnabled();

function createEmptyCounters() {
  return {
    startedAt: Date.now(),
    driverLiveMapRenders: 0,
    companyDashboardRenders: 0,
    fitBoundsCalls: 0,
    fitBoundsGpsBlocked: 0,
    markerCreates: 0,
    markerPositionUpdates: 0,
    invalidateQueries: 0,
    reactCommitMsTotal: 0,
    reactCommitCount: 0,
    gpsEvents: 0,
  };
}

let counters = createEmptyCounters();
let baselineActive = false;

export function isCompanyDashboardPerfEnabled() {
  return readPerfEnabled() || PERF_ENABLED;
}

export function startCompanyDashboardPerfBaseline() {
  if (typeof window === 'undefined') return;
  counters = createEmptyCounters();
  baselineActive = true;
  window.__companyDashboardPerfBaseline = {
    startedAt: counters.startedAt,
    report: reportCompanyDashboardPerfBaseline,
    stop: stopCompanyDashboardPerfBaseline,
  };
}

export function stopCompanyDashboardPerfBaseline() {
  baselineActive = false;
}

export function reportCompanyDashboardPerfBaseline() {
  const elapsedMin = Math.max((Date.now() - counters.startedAt) / 60_000, 0.001);
  const report = {
    elapsedMinutes: elapsedMin,
    driverLiveMapRendersPerMin: counters.driverLiveMapRenders / elapsedMin,
    companyDashboardRendersPerMin: counters.companyDashboardRenders / elapsedMin,
    fitBoundsCallsPerMin: counters.fitBoundsCalls / elapsedMin,
    fitBoundsGpsBlockedPerMin: counters.fitBoundsGpsBlocked / elapsedMin,
    invalidateQueriesPerMin: counters.invalidateQueries / elapsedMin,
    gpsEventsPerMin: counters.gpsEvents / elapsedMin,
    avgReactCommitMs:
      counters.reactCommitCount > 0
        ? counters.reactCommitMsTotal / counters.reactCommitCount
        : 0,
    markerCreates: counters.markerCreates,
    markerPositionUpdates: counters.markerPositionUpdates,
  };
  if (typeof window !== 'undefined') {
    window.__companyDashboardPerfLastReport = report;
  }
  if (process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.info('[CompanyDashboardPerf]', report);
  }
  return report;
}

function bump(field, delta = 1) {
  if (!readPerfEnabled() && !baselineActive) return;
  counters[field] = (counters[field] || 0) + delta;
}

export function recordDriverLiveMapRender() {
  bump('driverLiveMapRenders');
}

export function recordCompanyDashboardRender() {
  bump('companyDashboardRenders');
}

export function recordFitBoundsCall({ structural = true } = {}) {
  bump('fitBoundsCalls');
  if (!structural) bump('fitBoundsGpsBlocked');
}

export function recordMarkerCreate() {
  bump('markerCreates');
}

export function recordMarkerPositionUpdate() {
  bump('markerPositionUpdates');
}

export function recordGpsEvent() {
  bump('gpsEvents');
}

export function recordReactCommitMs(ms) {
  if (!readPerfEnabled() && !baselineActive) return;
  counters.reactCommitMsTotal += ms;
  counters.reactCommitCount += 1;
}

export function recordInvalidateQuery() {
  bump('invalidateQueries');
}

/** Wrapper optionnel autour de queryClient.invalidateQueries */
export function wrapInvalidateQueries(queryClient) {
  if (!queryClient || typeof queryClient.invalidateQueries !== 'function') return queryClient;
  if (queryClient.__companyPerfWrapped) return queryClient;
  const original = queryClient.invalidateQueries.bind(queryClient);
  queryClient.invalidateQueries = (...args) => {
    recordInvalidateQuery();
    return original(...args);
  };
  queryClient.__companyPerfWrapped = true;
  return queryClient;
}
