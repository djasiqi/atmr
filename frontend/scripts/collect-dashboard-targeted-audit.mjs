/**
 * Collecte audit ciblé dashboard (Bundle / Maps / Duplication + CWV).
 *
 * Usage:
 *   node scripts/collect-dashboard-targeted-audit.mjs --mode=dev --base-url=http://localhost:3000
 *   node scripts/collect-dashboard-targeted-audit.mjs --mode=prod-build --base-url=http://localhost:3000
 *
 * Prérequis: frontend démarré (dev ou serve -s build), compte démo company1@demo.lirie.ch
 */

import { chromium } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');
const RUNS_DIR = path.join(REPO_ROOT, 'docs/perf/runs');

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
const waitMs = Number(args.wait || 15000);

function stamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

async function collect() {
  fs.mkdirSync(RUNS_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  await page.goto(`${baseUrl}/login`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => sessionStorage.setItem('COMPANY_DASH_PERF', '1'));
  await page.locator('input[name="email"]').fill(email);
  await page.locator('input[name="password"]').fill(password);
  await page.getByRole('button', { name: /se connecter/i }).click();
  await page.waitForURL(/\/dashboard\/company\//, { timeout: 60000 });

  const match = page.url().match(/\/dashboard\/company\/([^/?#]+)/);
  const publicId = match?.[1];
  if (!publicId) throw new Error('public_id entreprise introuvable après login');

  const dashUrl = `${baseUrl}/dashboard/company/${publicId}?live_map=1`;
  await page.goto(dashUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForTimeout(waitMs);

  const payload = await page.evaluate(() => ({
    bundle: window.__companyDashboardBundleReport ?? null,
    maps: window.__companyDashboardMapsReport ?? null,
    duplication: window.__companyDashboardDuplicationReport ?? null,
    measures: window.__companyDashboardAuditReports?.report1?.webPerfMeasures
      ?? window.__companyDashboardAuditReports?.targeted?.maps?.webPerfMeasures
      ?? null,
    report1: window.__companyDashboardAuditReports?.report1 ?? null,
    webVitals: window.__companyDashboardWebVitals ?? null,
    context: {
      mode: null,
      userAgent: navigator.userAgent,
      timestamp: Date.now(),
      dashStartSource: window.__companyDashboardDashStartSource
        ?? window.__companyDashboardAuditReports?.dashStartSource
        ?? window.__companyDashboardAuditReports?.report1?.dashStartSource
        ?? null,
      url: window.location.href,
    },
  }));

  payload.context.mode = mode;

  const outFile = path.join(RUNS_DIR, `targeted-audit-${stamp()}-${mode}.json`);
  fs.writeFileSync(outFile, JSON.stringify(payload, null, 2), 'utf8');
  // eslint-disable-next-line no-console
  console.log(`[targeted-audit] écrit ${outFile}`);
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        dashStartSource: payload.context.dashStartSource,
        shellMs: payload.report1?.webPerfMeasures?.dashboard_shell_visible_ms,
        criticalMs: payload.report1?.webPerfMeasures?.dashboard_critical_ready_ms,
        lcp: payload.webVitals?.lcp,
        sdkNetworkMs: payload.maps?.sdkNetworkMs,
        dispatchMode: payload.duplication?.perKey?.dispatch_mode,
        alerts: payload.duplication?.perKey?.alerts,
        bundleTop3: (payload.bundle?.top20 || []).slice(0, 3).map((r) => ({
          name: r.name?.split('/').pop(),
          transferSize: r.transferSize,
        })),
      },
      null,
      2
    )
  );

  await browser.close();
  return outFile;
}

collect().catch((err) => {
  console.error(err);
  process.exit(1);
});
