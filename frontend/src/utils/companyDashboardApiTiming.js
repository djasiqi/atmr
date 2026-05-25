/**
 * Intercepteur axios — timing + taille payload pour audit dashboard entreprise.
 */

import apiClient from './apiClient';
import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';

const DASHBOARD_URL_PATTERNS = [
  { key: 'company_profile', re: /\/companies\/me$/ },
  { key: 'reservations', re: /\/companies\/me\/reservations/ },
  { key: 'drivers', re: /\/companies\/me\/drivers/ },
  { key: 'pending_window', re: /\/companies\/me\/reservations.*tab=pending|company-pending/ },
  { key: 'assignments', re: /\/company_dispatch\/assignments/ },
  { key: 'delays', re: /\/company_dispatch\/delays/ },
  { key: 'realtime_dashboard', re: /\/dashboard\/realtime/ },
  { key: 'dispatch_mode', re: /\/company_dispatch\/mode/ },
  { key: 'institution_offers', re: /\/request-offers/ },
  { key: 'messages', re: /\/messages\// },
  { key: 'alerts', re: /\/companies\/notifications/ },
];

function classifyUrl(url = '') {
  const path = String(url).split('?')[0];
  for (const { key, re } of DASHBOARD_URL_PATTERNS) {
    if (re.test(path) || re.test(url)) return key;
  }
  return null;
}

function estimatePayloadBytes(data) {
  if (data == null) return 0;
  try {
    if (typeof data === 'string') return new Blob([data]).size;
    return new Blob([JSON.stringify(data)]).size;
  } catch {
    return 0;
  }
}

function getStore() {
  if (typeof window === 'undefined') return { requests: [] };
  if (!window.__companyDashboardApiTiming) {
    window.__companyDashboardApiTiming = { requests: [], startedAt: Date.now() };
  }
  return window.__companyDashboardApiTiming;
}

let interceptorInstalled = false;

export function installCompanyDashboardApiTiming() {
  if (interceptorInstalled || typeof window === 'undefined') return;
  interceptorInstalled = true;

  apiClient.interceptors.request.use((config) => {
    if (!isCompanyDashboardPerfEnabled()) return config;
    const key = classifyUrl(config.url || '');
    if (!key && !config.__forceDashPerf) return config;
    config.metadata = config.metadata || {};
    config.metadata.dashPerfKey = key || 'other';
    config.metadata.dashPerfStart = performance.now();
    return config;
  });

  apiClient.interceptors.response.use(
    (response) => {
      if (!isCompanyDashboardPerfEnabled()) return response;
      const meta = response.config?.metadata;
      if (!meta?.dashPerfStart) return response;
      const durationMs = Math.round(performance.now() - meta.dashPerfStart);
      const payloadBytes = estimatePayloadBytes(response.data);
      const ttfbMs =
        response.request?.performanceTiming != null
          ? null
          : null;
      const sqlCountHeader = response.headers?.['x-sql-query-count'];
      const sqlDurationHeader = response.headers?.['x-sql-duration-ms'];
      const entry = {
        key: meta.dashPerfKey,
        url: response.config?.url,
        method: (response.config?.method || 'get').toUpperCase(),
        status: response.status,
        durationMs,
        payloadBytes,
        ttfbMs,
        at: Date.now(),
      };
      if (sqlCountHeader != null) {
        entry.sqlQueryCount = Number(sqlCountHeader);
        entry.sqlDurationMs = Number(sqlDurationHeader) || 0;
        if (typeof window !== 'undefined') {
          if (!window.__companyDashboardSqlPerf) {
            window.__companyDashboardSqlPerf = { endpoints: [] };
          }
          window.__companyDashboardSqlPerf.endpoints.push({
            key: meta.dashPerfKey,
            url: response.config?.url,
            sqlQueryCount: entry.sqlQueryCount,
            sqlDurationMs: entry.sqlDurationMs,
            apiDurationMs: durationMs,
            at: Date.now(),
          });
        }
      }
      getStore().requests.push(entry);
      return response;
    },
    (error) => {
      if (isCompanyDashboardPerfEnabled() && error.config?.metadata?.dashPerfStart) {
        const durationMs = Math.round(performance.now() - error.config.metadata.dashPerfStart);
        getStore().requests.push({
          key: error.config.metadata.dashPerfKey,
          url: error.config?.url,
          method: (error.config?.method || 'get').toUpperCase(),
          status: error.response?.status ?? 0,
          durationMs,
          payloadBytes: estimatePayloadBytes(error.response?.data),
          error: true,
          at: Date.now(),
        });
      }
      return Promise.reject(error);
    }
  );
}

export function getCompanyDashboardApiTimingSnapshot() {
  const store = getStore();
  return {
    requests: [...(store.requests || [])],
    totalPayloadBytes: (store.requests || []).reduce((s, r) => s + (r.payloadBytes || 0), 0),
  };
}

export function resetCompanyDashboardApiTiming() {
  if (typeof window !== 'undefined') {
    window.__companyDashboardApiTiming = { requests: [], startedAt: Date.now() };
  }
}
