/**
 * Lot 0 — Baseline perf espace entreprise (plan verrouillé).
 *
 * Scénarios :
 *   cold   — première visite (contexte neuf, SW désactivé)
 *   warm-sw — revisite avec SW (si disponible)
 *   nav-hot — navigations sidebar après login (sans re-goto dashboard)
 *
 * Usage :
 *   node scripts/collect-company-space-baseline.mjs --mode=prod-build --runs=5
 *   node scripts/collect-company-space-baseline.mjs --scenario=cold --runs=30
 *
 * Prérequis : frontend build/servi, compte démo company1@demo.lirie.ch
 * Aucune identité réelle n’est écrite dans les sorties (email synthétique uniquement).
 */

import { chromium } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const RUNS_DIR = path.join(REPO_ROOT, 'docs/perf/runs');
const BASELINE_DIR = path.join(REPO_ROOT, 'docs/perf');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? 'true'];
  })
);

const mode = args.mode || 'dev';
const baseUrl = (args['base-url'] || 'http://localhost:3000').replace(/\/$/, '');
const email = args.email || 'company1@demo.lirie.ch';
const password = args.password || 'LirieDemo2024!';
const runs = Math.max(1, Number(args.runs || 5));
const scenarioFilter = args.scenario || 'all';
const waitCriticalMs = Number(args.wait || 20000);
const headless = args.headless !== 'false';

const SIDEBAR_PATHS = [
  'reservations',
  'drivers',
  'clients',
  'invoices/clients',
  'dispatch',
  'settings',
  'analytics',
];

function stamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function percentile(sorted, p) {
  if (!sorted.length) return null;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function summarizeNumeric(values) {
  const nums = values.filter((v) => typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b);
  if (!nums.length) return null;
  return {
    n: nums.length,
    min: nums[0],
    p50: percentile(nums, 50),
    p75: percentile(nums, 75),
    p95: percentile(nums, 95),
    max: nums[nums.length - 1],
  };
}

/**
 * Collecte réseau sans PII : pas d’URL query complète si elle contient email/token.
 */
function sanitizeUrl(url) {
  try {
    const u = new URL(url);
    u.search = '';
    u.hash = '';
    return u.toString();
  } catch {
    return String(url || '').split('?')[0];
  }
}

async function loginAndGetPublicId(page) {
  await page.goto(`${baseUrl}/login`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    sessionStorage.setItem('COMPANY_DASH_PERF', '1');
  });
  await page.locator('input[name="email"]').fill(email);
  await page.locator('input[name="password"]').fill(password);
  await Promise.all([
    page.waitForURL(/\/dashboard\/company\//, { timeout: 90000 }),
    page.getByRole('button', { name: /se connecter/i }).click(),
  ]);
  const match = page.url().match(/\/dashboard\/company\/([^/?#]+)/);
  if (!match?.[1]) throw new Error('public_id entreprise introuvable après login');
  return match[1];
}

async function waitCriticalReady(page) {
  await page.waitForFunction(
    () => {
      const m =
        window.__companyDashboardAuditReports?.report1?.webPerfMeasures ||
        window.__companyDashboardWebVitals ||
        {};
      return (
        typeof m.dashboard_critical_ready_ms === 'number' ||
        document.querySelector('[data-testid="dashboard-critical-ready"]') != null ||
        document.querySelector('[data-tour-id="company-dashboard"]') != null
      );
    },
    { timeout: waitCriticalMs }
  ).catch(() => {});
  await page.waitForTimeout(1500);
}

async function collectPageMetrics(page, label) {
  const network = [];
  const onResponse = async (response) => {
    try {
      const req = response.request();
      if (req.resourceType() === 'xhr' || req.resourceType() === 'fetch' || req.resourceType() === 'document' || req.resourceType() === 'script' || req.resourceType() === 'stylesheet') {
        const headers = response.headers();
        const len = headers['content-length'] ? Number(headers['content-length']) : null;
        network.push({
          url: sanitizeUrl(response.url()),
          method: req.method(),
          status: response.status(),
          resourceType: req.resourceType(),
          transferSize: Number.isFinite(len) ? len : null,
        });
      }
    } catch {
      /* ignore */
    }
  };
  page.on('response', onResponse);

  const started = Date.now();
  await waitCriticalReady(page);
  const elapsedMs = Date.now() - started;

  page.off('response', onResponse);

  const pageEval = await page.evaluate(() => {
    const measures =
      window.__companyDashboardAuditReports?.report1?.webPerfMeasures ||
      {};
    const nav = performance.getEntriesByType('navigation')[0];
    const resources = performance.getEntriesByType('resource') || [];
    let jsTransfer = 0;
    let cssTransfer = 0;
    let longTasks = 0;
    let longTasksDuration = 0;
    try {
      const lts = performance.getEntriesByType('longtask') || [];
      longTasks = lts.length;
      longTasksDuration = lts.reduce((s, t) => s + (t.duration || 0), 0);
    } catch {
      /* longtask may be unavailable */
    }
    for (const r of resources) {
      const name = r.name || '';
      const size = r.transferSize || 0;
      if (/\.js(\?|$)/i.test(name) || r.initiatorType === 'script') jsTransfer += size;
      if (/\.css(\?|$)/i.test(name) || r.initiatorType === 'link') cssTransfer += size;
    }
    return {
      shellMs: measures.dashboard_shell_visible_ms ?? null,
      criticalMs: measures.dashboard_critical_ready_ms ?? null,
      webVitals: window.__companyDashboardWebVitals ?? null,
      jsTransfer,
      cssTransfer,
      longTasks,
      longTasksDuration,
      domContentLoaded: nav?.domContentLoadedEventEnd ?? null,
      transferSizeNav: nav?.transferSize ?? null,
      urlPath: window.location.pathname,
    };
  });

  const getCount = network.filter(
    (n) => n.method === 'GET' && (n.resourceType === 'xhr' || n.resourceType === 'fetch')
  ).length;
  const transferSum = network.reduce((s, n) => s + (n.transferSize || 0), 0);

  return {
    label,
    elapsedMs,
    getCountXhrFetch: getCount,
    networkTransferSum: transferSum,
    ...pageEval,
    // Pas de body réseau ni query strings (PII).
    networkSampleTop: network
      .slice(0, 40)
      .map((n) => ({
        method: n.method,
        status: n.status,
        type: n.resourceType,
        path: (() => {
          try {
            return new URL(n.url).pathname;
          } catch {
            return n.url;
          }
        })(),
        transferSize: n.transferSize,
      })),
  };
}

async function runCold(browser) {
  const context = await browser.newContext({
    serviceWorkers: 'block',
  });
  const page = await context.newPage();
  const publicId = await loginAndGetPublicId(page);
  // Ne PAS re-goto : mesure sur la redirection naturelle post-login.
  const metrics = await collectPageMetrics(page, 'cold-post-login');
  await context.close();
  return { scenario: 'cold', publicIdPresent: Boolean(publicId), metrics };
}

async function runWarmSw(browser) {
  const context = await browser.newContext({
    serviceWorkers: 'allow',
  });
  const page = await context.newPage();
  await loginAndGetPublicId(page);
  await waitCriticalReady(page);
  // Revisite : reload pour laisser le SW intervenir.
  await page.reload({ waitUntil: 'domcontentloaded' });
  const metrics = await collectPageMetrics(page, 'warm-sw-reload');
  await context.close();
  return { scenario: 'warm-sw', metrics };
}

async function runNavHot(browser) {
  const context = await browser.newContext({ serviceWorkers: 'block' });
  const page = await context.newPage();
  const publicId = await loginAndGetPublicId(page);
  await waitCriticalReady(page);
  const navMetrics = [];
  for (const suffix of SIDEBAR_PATHS) {
    const t0 = Date.now();
    await page.goto(`${baseUrl}/dashboard/company/${publicId}/${suffix}`, {
      waitUntil: 'domcontentloaded',
      timeout: 60000,
    });
    await page.waitForTimeout(1200);
    navMetrics.push({
      path: suffix,
      elapsedMs: Date.now() - t0,
      urlPath: new URL(page.url()).pathname.replace(/\/[0-9a-f-]{20,}/i, '/:id'),
    });
  }
  await context.close();
  return { scenario: 'nav-hot', navMetrics };
}

async function main() {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  fs.mkdirSync(BASELINE_DIR, { recursive: true });

  const browser = await chromium.launch({ headless });
  const results = { cold: [], 'warm-sw': [], 'nav-hot': [] };
  const scenarios =
    scenarioFilter === 'all'
      ? ['cold', 'warm-sw', 'nav-hot']
      : [scenarioFilter];

  for (let i = 0; i < runs; i += 1) {
    // eslint-disable-next-line no-console
    console.log(`[baseline] run ${i + 1}/${runs}`);
    if (scenarios.includes('cold')) {
      results.cold.push(await runCold(browser));
    }
    if (scenarios.includes('warm-sw')) {
      results['warm-sw'].push(await runWarmSw(browser));
    }
    if (scenarios.includes('nav-hot')) {
      results['nav-hot'].push(await runNavHot(browser));
    }
  }

  await browser.close();

  const aggregate = {
    generatedAt: new Date().toISOString(),
    mode,
    baseUrlHost: (() => {
      try {
        return new URL(baseUrl).host;
      } catch {
        return 'unknown';
      }
    })(),
    runs,
    scenarios: {},
  };

  for (const key of ['cold', 'warm-sw']) {
    const list = results[key];
    if (!list.length) continue;
    aggregate.scenarios[key] = {
      getCountXhrFetch: summarizeNumeric(list.map((r) => r.metrics.getCountXhrFetch)),
      networkTransferSum: summarizeNumeric(list.map((r) => r.metrics.networkTransferSum)),
      jsTransfer: summarizeNumeric(list.map((r) => r.metrics.jsTransfer)),
      cssTransfer: summarizeNumeric(list.map((r) => r.metrics.cssTransfer)),
      shellMs: summarizeNumeric(list.map((r) => r.metrics.shellMs)),
      criticalMs: summarizeNumeric(list.map((r) => r.metrics.criticalMs)),
      longTasks: summarizeNumeric(list.map((r) => r.metrics.longTasks)),
      longTasksDuration: summarizeNumeric(list.map((r) => r.metrics.longTasksDuration)),
      lcp: summarizeNumeric(list.map((r) => r.metrics.webVitals?.lcp)),
    };
  }

  if (results['nav-hot'].length) {
    const byPath = {};
    for (const run of results['nav-hot']) {
      for (const n of run.navMetrics || []) {
        byPath[n.path] = byPath[n.path] || [];
        byPath[n.path].push(n.elapsedMs);
      }
    }
    aggregate.scenarios['nav-hot'] = Object.fromEntries(
      Object.entries(byPath).map(([p, vals]) => [p, summarizeNumeric(vals)])
    );
  }

  // Budgets provisoires (à recalibrer après ≥30 runs labo). Valeurs conservatrices.
  aggregate.proposedLabBudgets = {
    note: 'Provisoires jusqu’à n≥30. Gates de fusion labo — pas les CWV terrain.',
    maxGetCriticalUntilReady: 5,
    maxCriticalTransferBytes: 2_500_000,
    maxInitialJsCompressedBytes: 1_200_000,
    maxBootstrapResponseBytes: 350_000,
    maxApiP95Ms: 1500,
    maxLongTasksUntilReady: 25,
    maxLongTasksDurationMs: 3000,
    maxWorkboxManifestBytes: 1_500_000,
  };

  const outFile = path.join(RUNS_DIR, `company-space-baseline-${stamp()}-${mode}.json`);
  fs.writeFileSync(
    outFile,
    JSON.stringify({ aggregate, rawCounts: Object.fromEntries(Object.entries(results).map(([k, v]) => [k, v.length])) }, null, 2),
    'utf8'
  );
  // eslint-disable-next-line no-console
  console.log(`[baseline] écrit ${outFile}`);
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(aggregate, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
