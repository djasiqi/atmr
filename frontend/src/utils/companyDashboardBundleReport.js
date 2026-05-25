/**
 * Rapport bundle runtime — scripts téléchargés / prefetch au chargement dashboard.
 */

import { isCompanyDashboardPerfEnabled } from './companyDashboardPerfInstrumentation';
import {
  getResourceBufferDiagnostics,
  getStreamedResourceEntries,
} from './companyDashboardPerfBootstrap';

const PREFETCH_CHUNK_HINTS = [
  'CompanyInvoices',
  'ClientInvoices',
  'Invoices',
  'exceljs',
  'file-saver',
  'recharts',
  'react-icons',
];

function getDashStartTime() {
  if (typeof performance === 'undefined' || !performance.getEntriesByName) return 0;
  const marks = performance.getEntriesByName('dashboard_start', 'mark');
  return marks.length ? marks[marks.length - 1].startTime : 0;
}

function classifyChunk(entry) {
  const name = entry.name || '';
  if (entry.initiatorType === 'link' || name.includes('prefetch')) return 'prefetched';
  if (PREFETCH_CHUNK_HINTS.some((h) => name.includes(h))) return 'prefetched';
  const dashStart = getDashStartTime();
  if (dashStart && entry.startTime > dashStart + 8000) return 'lazy-on-demand';
  if (dashStart && entry.startTime <= dashStart + 3000) return 'critical';
  return 'other';
}

function collectPrefetchLinks() {
  if (typeof document === 'undefined') return [];
  return [...document.querySelectorAll('link[rel="prefetch"], link[rel="preload"]')].map(
    (el) => ({
      rel: el.getAttribute('rel'),
      href: el.getAttribute('href'),
      as: el.getAttribute('as'),
    })
  );
}

function mergeResourceEntries() {
  if (typeof performance === 'undefined') return [];
  const seen = new Set();
  const out = [];
  const push = (e) => {
    const key = `${e.name}|${e.startTime}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(e);
  };
  performance.getEntriesByType('resource').forEach(push);
  getStreamedResourceEntries().forEach(push);
  return out;
}

export function buildBundleReport() {
  if (!isCompanyDashboardPerfEnabled()) {
    return { enabled: false };
  }

  const dashStart = getDashStartTime();
  const windowEnd = dashStart ? dashStart + 15000 : Infinity;

  const resources = mergeResourceEntries()
    .filter((e) => {
      const isScript =
        e.initiatorType === 'script' ||
        (e.name && (e.name.endsWith('.js') || e.name.includes('.chunk.js')));
      if (!isScript) return false;
      if (dashStart && e.startTime > windowEnd) return false;
      return true;
    })
    .map((e) => ({
      name: e.name,
      transferSize: e.transferSize || 0,
      encodedBodySize: e.encodedBodySize || 0,
      decodedBodySize: e.decodedBodySize || 0,
      initiatorType: e.initiatorType,
      startTime: Math.round(e.startTime),
      responseEnd: Math.round(e.responseEnd),
      durationMs: Math.round(e.duration),
      nextHopProtocol: e.nextHopProtocol || null,
      loadClass: classifyChunk(e),
    }))
    .sort((a, b) => b.transferSize - a.transferSize);

  const totals = resources.reduce(
    (acc, r) => {
      acc.bytesDownloadedTotal += r.transferSize;
      if (r.loadClass === 'critical') acc.bytesCritical += r.transferSize;
      else if (r.loadClass === 'prefetched') acc.bytesPrefetched += r.transferSize;
      else if (r.loadClass === 'lazy-on-demand') acc.bytesLazy += r.transferSize;
      else acc.bytesOther += r.transferSize;
      return acc;
    },
    {
      bytesDownloadedTotal: 0,
      bytesCritical: 0,
      bytesPrefetched: 0,
      bytesLazy: 0,
      bytesOther: 0,
    }
  );

  const top20 = resources.slice(0, 20);
  const prefetchLinksAtPublish = collectPrefetchLinks();

  return {
    title: 'Rapport Bundle — scripts dashboard (runtime)',
    generatedAt: new Date().toISOString(),
    dashStartRelativeMs: dashStart ? 0 : null,
    observationWindowMs: 15000,
    ...totals,
    scriptResources: resources,
    top20,
    prefetchLinks: prefetchLinksAtPublish,
    bufferDiagnostics: getResourceBufferDiagnostics(),
    suspects: {
      reactIconsBytes: resources
        .filter((r) => r.name.includes('react-icons'))
        .reduce((s, r) => s + r.transferSize, 0),
      invoicesBytes: resources
        .filter((r) => /Invoices|invoices/i.test(r.name))
        .reduce((s, r) => s + r.transferSize, 0),
      exceljsBytes: resources
        .filter((r) => r.name.includes('exceljs'))
        .reduce((s, r) => s + r.transferSize, 0),
      rechartsBytes: resources
        .filter((r) => r.name.includes('recharts'))
        .reduce((s, r) => s + r.transferSize, 0),
    },
  };
}

export function publishBundleReport() {
  const report = buildBundleReport();
  if (typeof window !== 'undefined') {
    window.__companyDashboardBundleReport = report;
  }
  return report;
}
