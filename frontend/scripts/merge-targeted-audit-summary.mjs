/**
 * Fusionne deux runs dev + prod-build en docs/perf/targeted-audit-summary.md
 *
 * Usage:
 *   node scripts/merge-targeted-audit-summary.mjs --dev=docs/perf/runs/targeted-audit-XXX-dev.json --prod=docs/perf/runs/targeted-audit-YYY-prod-build.json
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v];
  })
);

function readJson(p) {
  const full = path.isAbsolute(p) ? p : path.join(REPO_ROOT, p);
  return JSON.parse(fs.readFileSync(full, 'utf8'));
}

function classifyDuplication(devReport, prodReport) {
  const keys = new Set([
    ...Object.keys(devReport?.duplication?.perKey || {}),
    ...Object.keys(prodReport?.duplication?.perKey || {}),
  ]);
  const out = {};
  keys.forEach((key) => {
    const devCount = devReport?.duplication?.perKey?.[key]?.count ?? 0;
    const prodCount = prodReport?.duplication?.perKey?.[key]?.count ?? 0;
    let cause = 'unknown';
    if (devCount >= 2 && prodCount <= 1) cause = 'strict_mode';
    else if (devCount >= 2 && prodCount >= 2) cause = 'true_duplicate';
    else if (devCount <= 1) cause = 'single';
    out[key] = { devCount, prodCount, cause };
  });
  return out;
}

function fmtMs(v) {
  if (v == null) return '—';
  return `${v} ms`;
}

function main() {
  const devPath = args.dev;
  const prodPath = args.prod;
  if (!devPath || !prodPath) {
    console.error('Usage: --dev=<json> --prod=<json>');
    process.exit(1);
  }
  const dev = readJson(devPath);
  const prod = readJson(prodPath);
  const dup = classifyDuplication(dev, prod);

  const lines = [
    '# Résumé audit ciblé — Bundle / Maps / Duplication',
    '',
    `Généré: ${new Date().toISOString()}`,
    '',
    '## Core Web Vitals',
    '',
    '| Metric | dev | prod-build | Budget |',
    '|--------|-----|------------|--------|',
    `| LCP | ${fmtMs(dev.webVitals?.lcp)} | ${fmtMs(prod.webVitals?.lcp)} | 2500 ms |`,
    `| INP | ${fmtMs(dev.webVitals?.inp)} | ${fmtMs(prod.webVitals?.inp)} | 200 ms |`,
    `| CLS | ${dev.webVitals?.cls ?? '—'} | ${prod.webVitals?.cls ?? '—'} | 0.10 |`,
    '',
    '## Dashboard marks',
    '',
    `| Mark | dev | prod |`,
    `| dashboard_shell_visible_ms | ${dev.report1?.webPerfMeasures?.dashboard_shell_visible_ms ?? '—'} | ${prod.report1?.webPerfMeasures?.dashboard_shell_visible_ms ?? '—'} |`,
    `| dashboard_critical_ready_ms | ${dev.report1?.webPerfMeasures?.dashboard_critical_ready_ms ?? '—'} | ${prod.report1?.webPerfMeasures?.dashboard_critical_ready_ms ?? '—'} |`,
    `| dashStartSource | ${dev.context?.dashStartSource ?? '—'} | ${prod.context?.dashStartSource ?? '—'} |`,
    '',
    '## Maps',
    '',
    `| Phase | dev | prod |`,
    `| SDK réseau | ${fmtMs(dev.maps?.sdkNetworkMs)} | ${fmtMs(prod.maps?.sdkNetworkMs)} |`,
    `| Construction carte | ${fmtMs(dev.maps?.mapConstructMs)} | ${fmtMs(prod.maps?.mapConstructMs)} |`,
    `| Markers après map | ${fmtMs(dev.maps?.markersAfterMapMs)} | ${fmtMs(prod.maps?.markersAfterMapMs)} |`,
    `| markerCount | ${dev.maps?.markerCount ?? '—'} | ${prod.maps?.markerCount ?? '—'} |`,
    '',
    '## Duplication (dev vs prod)',
    '',
    '| key | dev count | prod count | cause |',
    '|-----|-----------|------------|-------|',
  ];

  Object.entries(dup).forEach(([key, v]) => {
    lines.push(`| ${key} | ${v.devCount} | ${v.prodCount} | ${v.cause} |`);
  });

  lines.push('', '## Bundle (top 5 transferSize dev)', '');
  (dev.bundle?.top20 || []).slice(0, 5).forEach((r, i) => {
    lines.push(`${i + 1}. ${r.name?.split('/').pop()} — ${r.transferSize} B (${r.loadClass})`);
  });

  const outPath = path.join(REPO_ROOT, 'docs/perf/targeted-audit-summary.md');
  fs.writeFileSync(outPath, lines.join('\n'), 'utf8');
  console.log(`[merge] ${outPath}`);
}

main();
