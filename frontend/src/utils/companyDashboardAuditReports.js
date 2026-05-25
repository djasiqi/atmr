/**
 * Génération des 5 rapports chiffrés — audit perf dashboard entreprise.
 */

import {
  isCompanyDashboardPerfEnabled,
  reportCompanyDashboardPerfBaseline,
} from './companyDashboardPerfInstrumentation';
import { runDashboardPerfMeasures, getCompanyDashboardWebVitalsSnapshot } from './companyDashboardWebPerf';
import { getCompanyDashboardApiTimingSnapshot } from './companyDashboardApiTiming';
import { publishBundleReport } from './companyDashboardBundleReport';
import { publishMapsReport } from './companyDashboardMapsReport';
import { publishDuplicationReport } from './companyDashboardDuplicationReport';

export const ENDPOINT_BUDGET_MS = {
  company_profile: 300,
  reservations: 700,
  drivers: 400,
  pending_window: 500,
  messages: 400,
  alerts: 400,
  assignments: 700,
  delays: 500,
  realtime_dashboard: 600,
  dispatch_mode: 300,
  institution_offers: 500,
};

export const PAYLOAD_BUDGET_BYTES = 300 * 1024;

export const CWV_BUDGET = { lcp: 2500, inp: 200, cls: 0.1 };

export const SQL_BUDGET = { maxQueries: 20 };

/** Classification requêtes cold-load dashboard */
export const REQUEST_LOAD_CLASS = {
  blocking: [
    'company_profile',
    'reservations',
    'drivers',
    'assignments',
    'delays',
    'realtime_dashboard',
  ],
  non_blocking: ['pending_window', 'institution_offers', 'dispatch_mode', 'alerts'],
  deferred_lazy: ['messages', 'gmaps_sdk', 'gmaps_map'],
};

function flagBudget(actual, budget, higherIsWorse = true) {
  if (actual == null || budget == null) return 'unknown';
  if (higherIsWorse) return actual <= budget ? 'ok' : 'P0 regression candidate';
  return actual >= budget ? 'ok' : 'P0 regression candidate';
}

function aggregateApiByKey(requests) {
  const byKey = {};
  for (const r of requests) {
    const k = r.key || 'other';
    if (!byKey[k]) byKey[k] = { count: 0, maxDurationMs: 0, totalPayloadBytes: 0, urls: [] };
    byKey[k].count += 1;
    byKey[k].maxDurationMs = Math.max(byKey[k].maxDurationMs, r.durationMs || 0);
    byKey[k].totalPayloadBytes += r.payloadBytes || 0;
    if (r.url) byKey[k].urls.push(r.url);
  }
  return byKey;
}

export function buildAuditReport1() {
  const measures = runDashboardPerfMeasures();
  const cwv = getCompanyDashboardWebVitalsSnapshot();
  const api = getCompanyDashboardApiTimingSnapshot();
  const dashStartSource =
    typeof window !== 'undefined' ? window.__companyDashboardDashStartSource : null;
  const totalMs = measures.dashboard_critical_ready_ms ?? measures.dashboard_first_render_ms ?? null;
  return {
    title: 'Rapport 1 — Temps total + split API / React / Maps + CWV',
    generatedAt: new Date().toISOString(),
    dashStartSource: dashStartSource || 'unknown',
    dashboardTotalMs: totalMs,
    webPerfMeasures: measures,
    coreWebVitals: {
      ...cwv,
      budgets: CWV_BUDGET,
      lcpStatus: flagBudget(cwv.lcp, CWV_BUDGET.lcp),
      inpStatus: flagBudget(cwv.inp, CWV_BUDGET.inp),
      clsStatus: flagBudget(cwv.cls, CWV_BUDGET.cls),
    },
    splitEstimate: {
      apiMs: api.requests.reduce((m, r) => Math.max(m, r.durationMs || 0), 0),
      mapsMs: measures.gmaps_sdk_load_ms ?? null,
      shellToCriticalMs: measures.dashboard_critical_ready_ms ?? null,
    },
    hypothesis: [
      'Cas A: Google Maps 2–3s + payload lourd',
      'Cas B: waterfall profile → reservations → drivers',
      'Cas C: N+1 SQL sur booking.serialize',
      'Cas D: socket → invalidate → refetch avant stabilisation',
    ],
  };
}

export function buildAuditReport2() {
  const baseline = reportCompanyDashboardPerfBaseline();
  return {
    title: 'Rapport 2 — Top composants (renders / commit)',
    generatedAt: new Date().toISOString(),
    topComponents: [
      { name: 'CompanyDashboard', rendersPerMin: baseline.companyDashboardRendersPerMin },
      { name: 'DriverLiveMap', rendersPerMin: baseline.driverLiveMapRendersPerMin },
    ],
    fitBoundsCallsPerMin: baseline.fitBoundsCallsPerMin,
    avgReactCommitMs: baseline.avgReactCommitMs,
    markerCreates: baseline.markerCreates,
    markerPositionUpdates: baseline.markerPositionUpdates,
  };
}

export function buildAuditReport3() {
  const api = getCompanyDashboardApiTimingSnapshot();
  const byKey = aggregateApiByKey(api.requests);
  const endpoints = Object.entries(byKey)
    .map(([key, stats]) => ({
      key,
      ...stats,
      budgetMs: ENDPOINT_BUDGET_MS[key] ?? null,
      status: flagBudget(stats.maxDurationMs, ENDPOINT_BUDGET_MS[key]),
      loadClass: REQUEST_LOAD_CLASS.blocking.includes(key)
        ? 'blocking'
        : REQUEST_LOAD_CLASS.non_blocking.includes(key)
          ? 'non_blocking'
          : REQUEST_LOAD_CLASS.deferred_lazy.includes(key)
            ? 'deferred_lazy'
            : 'other',
    }))
    .sort((a, b) => b.maxDurationMs - a.maxDurationMs);

  const principalPayload =
    (byKey.reservations?.totalPayloadBytes || 0) + (byKey.drivers?.totalPayloadBytes || 0);

  return {
    title: 'Rapport 3 — Top endpoints (temps, taille, budgets)',
    generatedAt: new Date().toISOString(),
    top10: endpoints.slice(0, 10),
    payloadDashboardPrincipalBytes: principalPayload,
    payloadBudgetBytes: PAYLOAD_BUDGET_BYTES,
    payloadStatus: flagBudget(principalPayload, PAYLOAD_BUDGET_BYTES),
    requestLoadClassification: REQUEST_LOAD_CLASS,
  };
}

export function buildAuditReport4() {
  const baseline = reportCompanyDashboardPerfBaseline();
  return {
    title: 'Rapport 4 — Cycle temps réel (socket → invalidate → refetch)',
    generatedAt: new Date().toISOString(),
    invalidateQueriesPerMin: baseline.invalidateQueriesPerMin,
    gpsEventsPerMin: baseline.gpsEventsPerMin,
    cycle: 'socket_event → invalidateQueries → refetch → rerender',
    recommendation:
      'Grace period 3s après mount + invalidations ciblées (éviter refetchAll global)',
  };
}

export function buildAuditReport5() {
  const nav =
    typeof performance !== 'undefined' && performance.getEntriesByType
      ? performance.getEntriesByType('resource').filter((e) => e.initiatorType === 'script')
      : [];
  const scripts = nav
    .map((e) => ({
      name: e.name,
      transferSize: e.transferSize || 0,
      duration: Math.round(e.duration),
    }))
    .sort((a, b) => b.transferSize - a.transferSize)
    .slice(0, 15);

  const gmaps = nav.filter((e) => e.name.includes('maps.googleapis.com'));
  const chunks = nav.filter((e) => e.name.includes('.js') && !e.name.includes('maps.googleapis'));

  return {
    title: 'Rapport 5 — Bundle JavaScript dashboard',
    generatedAt: new Date().toISOString(),
    scriptResourcesTop: scripts,
    googleMapsScripts: gmaps.map((e) => ({
      name: e.name,
      transferSize: e.transferSize,
      duration: Math.round(e.duration),
    })),
    dashboardChunksSample: chunks.slice(0, 10),
    note: 'Utiliser Chrome Coverage + build analyzer pour JS exécuté vs téléchargé',
  };
}

export function buildAuditReportSql(backendSnapshot = {}) {
  return {
    title: 'Rapport 3b — SQL backend (endpoints critiques)',
    generatedAt: new Date().toISOString(),
    endpoints: backendSnapshot.endpoints || [],
    budget: SQL_BUDGET,
    note: 'Headers X-SQL-Query-Count / X-SQL-Duration-Ms quand COMPANY_DASH_PERF_SQL=1',
  };
}

export function buildAllAuditReports(backendSql = {}) {
  return {
    report1: buildAuditReport1(),
    report2: buildAuditReport2(),
    report3: buildAuditReport3(),
    report4: buildAuditReport4(),
    report5: buildAuditReport5(),
    reportSql: buildAuditReportSql(backendSql),
  };
}

export function publishTargetedAuditReports(backendSql = {}) {
  const reports = buildAllAuditReports(backendSql);
  const bundle = publishBundleReport();
  const maps = publishMapsReport();
  const duplication = publishDuplicationReport();
  const payload = {
    ...reports,
    targeted: { bundle, maps, duplication },
    dashStartSource:
      typeof window !== 'undefined' ? window.__companyDashboardDashStartSource : null,
    webVitals: getCompanyDashboardWebVitalsSnapshot(),
  };
  if (typeof window !== 'undefined') {
    window.__companyDashboardAuditReports = payload;
    window.__companyDashboardBundleReport = bundle;
    window.__companyDashboardMapsReport = maps;
    window.__companyDashboardDuplicationReport = duplication;
  }
  return payload;
}

export function publishAuditReports(backendSql = {}) {
  if (!isCompanyDashboardPerfEnabled() && typeof window === 'undefined') return null;
  const reports = publishTargetedAuditReports(backendSql);
  if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.info('[CompanyDashboardAudit]', reports);
  }
  return reports;
}

export function scheduleAuditReportPublish(delayMs = 8000) {
  if (typeof window === 'undefined') return () => {};
  const id = window.setTimeout(() => {
    const sqlSnapshot =
      typeof window !== 'undefined' ? window.__companyDashboardSqlPerf : {};
    publishAuditReports(sqlSnapshot || {});
  }, delayMs);
  return () => clearTimeout(id);
}
