/**
 * Web Performance API + Core Web Vitals pour l'audit dashboard entreprise.
 */

import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';

const MEASURE_PAIRS = [
  ['dashboard_start', 'dashboard_shell_visible', 'dashboard_shell_visible_ms'],
  ['dashboard_start', 'dashboard_critical_ready', 'dashboard_critical_ready_ms'],
  ['dashboard_start', 'dashboard_queries_loaded', 'dashboard_queries_loaded_ms'],
  ['dashboard_start', 'dashboard_first_render', 'dashboard_first_render_ms'],
  ['dashboard_start', 'dashboard_live_map_enabled', 'dashboard_live_map_enabled_ms'],
  ['gmaps_sdk_start', 'gmaps_sdk_end', 'gmaps_sdk_load_ms'],
  ['gmaps_sdk_request_start', 'gmaps_sdk_loaded', 'gmaps_sdk_request_load_ms'],
  ['gmaps_provider_mount', 'gmaps_map_loaded', 'gmaps_provider_to_map_ms'],
  ['gmaps_map_constructor_start', 'gmaps_map_loaded', 'gmaps_map_construct_ms'],
  ['dashboard_live_map_enabled', 'gmaps_map_loaded', 'gmaps_map_after_enable_ms'],
  ['gmaps_map_loaded', 'gmaps_first_markers', 'gmaps_first_markers_ms'],
];

export function perfMark(name) {
  if (typeof performance === 'undefined' || !performance.mark) return;
  if (!isCompanyDashboardPerfEnabled()) return;
  try {
    performance.mark(name);
  } catch {
    // ignore duplicate or invalid marks
  }
}

export function perfMeasure(name, startMark, endMark) {
  if (typeof performance === 'undefined' || !performance.measure) return null;
  if (!isCompanyDashboardPerfEnabled()) return null;
  try {
    performance.measure(name, startMark, endMark);
    const entries = performance.getEntriesByName(name, 'measure');
    return entries.length ? entries[entries.length - 1].duration : null;
  } catch {
    return null;
  }
}

export function runDashboardPerfMeasures() {
  if (!isCompanyDashboardPerfEnabled()) return {};
  const out = {};
  MEASURE_PAIRS.forEach(([start, end, key]) => {
    const duration = perfMeasure(key, start, end);
    if (duration != null) out[key] = Math.round(duration);
  });
  if (out.gmaps_sdk_request_load_ms == null && out.gmaps_sdk_load_ms != null) {
    out.gmaps_sdk_request_load_ms = out.gmaps_sdk_load_ms;
  }
  return out;
}

let cwvObserversStarted = false;

export function startCompanyDashboardWebVitals() {
  if (typeof window === 'undefined' || cwvObserversStarted) return;
  if (!isCompanyDashboardPerfEnabled()) return;
  cwvObserversStarted = true;

  const store = {
    lcp: null,
    inp: null,
    cls: 0,
    updatedAt: Date.now(),
  };
  window.__companyDashboardWebVitals = store;

  try {
    if (typeof PerformanceObserver !== 'undefined') {
      const lcpObs = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const last = entries[entries.length - 1];
        if (last) {
          store.lcp = Math.round(last.startTime);
          store.updatedAt = Date.now();
        }
      });
      lcpObs.observe({ type: 'largest-contentful-paint', buffered: true });

      const clsObs = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!entry.hadRecentInput) {
            store.cls = (store.cls || 0) + (entry.value || 0);
          }
        }
        store.updatedAt = Date.now();
      });
      clsObs.observe({ type: 'layout-shift', buffered: true });

      const inpObs = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const duration = entry.duration ?? entry.processingStart - entry.startTime;
          if (duration != null) {
            store.inp = Math.max(store.inp ?? 0, Math.round(duration));
            store.updatedAt = Date.now();
          }
        }
      });
      try {
        inpObs.observe({ type: 'event', buffered: true, durationThreshold: 0 });
      } catch {
        inpObs.observe({ type: 'first-input', buffered: true });
      }
    }
  } catch {
    // observers optional
  }
}

export function getCompanyDashboardWebVitalsSnapshot() {
  if (typeof window === 'undefined') return { lcp: null, inp: null, cls: null };
  const store = window.__companyDashboardWebVitals || {};
  return {
    lcp: store.lcp ?? null,
    inp: store.inp ?? null,
    cls: store.cls != null ? Number(store.cls.toFixed(4)) : null,
  };
}
