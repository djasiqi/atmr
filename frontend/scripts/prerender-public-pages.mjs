/**
 * Pré-rendu déterministe des pages publiques LIRIE (SEO-01B).
 * Serveur local sur build/ ; aucune dépendance à l’API prod / Maps / Nominatim.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';
import { validatePrerenderedHtml } from './validate-prerendered-html.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(FRONTEND_ROOT, 'build');

const PUBLIC_ROUTES = [
  '/',
  '/deplacez-vous',
  '/conduire',
  '/professionnel',
  '/a-propos',
  '/aide',
  '/contact',
  '/privacy',
  '/conditions',
  '/mentions-legales',
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.map': 'application/json',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};

function resolveBuildFile(urlPath) {
  const clean = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  const normalized = clean === '/' ? '/index.html' : clean;

  const candidates = [];
  if (normalized.endsWith('/')) {
    candidates.push(path.join(BUILD_DIR, normalized, 'index.html'));
  } else {
    candidates.push(path.join(BUILD_DIR, normalized));
    candidates.push(path.join(BUILD_DIR, `${normalized}.html`));
    candidates.push(path.join(BUILD_DIR, normalized, 'index.html'));
  }

  for (const candidate of candidates) {
    const resolved = path.resolve(candidate);
    if (!resolved.startsWith(BUILD_DIR)) continue;
    if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
      return resolved;
    }
  }

  // Pendant le premier passage : SPA shell pour laisser React router.
  return path.join(BUILD_DIR, 'index.html');
}

function startStaticServer() {
  const server = http.createServer((req, res) => {
    try {
      const filePath = resolveBuildFile(req.url || '/');
      const ext = path.extname(filePath).toLowerCase();
      const body = fs.readFileSync(filePath);
      res.writeHead(200, {
        'Content-Type': MIME[ext] || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      res.end(body);
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(`Erreur serveur pré-rendu: ${err.message}`);
    }
  });

  return new Promise((resolve, reject) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        port,
        origin: `http://127.0.0.1:${port}`,
        close: () =>
          new Promise((resClose, rejClose) => {
            server.close((err) => (err ? rejClose(err) : resClose()));
          }),
      });
    });
    server.on('error', reject);
  });
}

function outputPathForRoute(route) {
  if (route === '/') {
    return path.join(BUILD_DIR, 'index.html');
  }
  const dir = path.join(BUILD_DIR, route.replace(/^\//, ''));
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, 'index.html');
}

function dedupeHeadTags(html) {
  // Une seule balise robots / description / canonical / title / JSON-LD.
  const keepLastMeta = (source, attrName, attrValue) => {
    const re = new RegExp(
      `<meta[^>]*${attrName}=["']${attrValue}["'][^>]*>`,
      'gi'
    );
    const matches = source.match(re) || [];
    if (matches.length <= 1) return source;
    let seen = 0;
    return source.replace(re, (m) => {
      seen += 1;
      return seen === matches.length ? m : '';
    });
  };

  let out = html;
  out = keepLastMeta(out, 'name', 'robots');
  out = keepLastMeta(out, 'name', 'description');

  const canonicalRe = /<link[^>]*rel=["']canonical["'][^>]*>/gi;
  const canonicals = out.match(canonicalRe) || [];
  if (canonicals.length > 1) {
    let seen = 0;
    out = out.replace(canonicalRe, (m) => {
      seen += 1;
      return seen === canonicals.length ? m : '';
    });
  }

  const titleRe = /<title[^>]*>[\s\S]*?<\/title>/gi;
  const titles = out.match(titleRe) || [];
  if (titles.length > 1) {
    let seen = 0;
    out = out.replace(titleRe, (m) => {
      seen += 1;
      return seen === titles.length ? m : '';
    });
  }

  const ldRe =
    /<script[^>]*type=["']application\/ld\+json["'][^>]*>[\s\S]*?<\/script>/gi;
  const lds = out.match(ldRe) || [];
  if (lds.length > 1) {
    let seen = 0;
    out = out.replace(ldRe, (m) => {
      seen += 1;
      return seen === lds.length ? m : '';
    });
  }

  return out;
}

async function configurePage(page) {
  await page.addInitScript(() => {
    try {
      localStorage.clear();
      sessionStorage.clear();
    } catch (_) {
      /* ignore */
    }
    Object.defineProperty(navigator, 'geolocation', {
      configurable: true,
      value: {
        getCurrentPosition: (_ok, err) => {
          if (typeof err === 'function') {
            err({ code: 1, message: 'Permission denied (prerender)' });
          }
        },
        watchPosition: () => 0,
        clearWatch: () => {},
      },
    });
  });

  await page.route('**/maps.googleapis.com/**', (route) => route.abort());
  await page.route('**/maps.gstatic.com/**', (route) => route.abort());
  await page.route('**/nominatim.openstreetmap.org/**', (route) => route.abort());

  await page.route('**/api/v1/public/platform-stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        completedBookings: 0,
        activeCompanies: 0,
        activeInstitutions: 0,
      }),
    });
  });
  await page.route('**/public/platform-stats**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        completedBookings: 0,
        activeCompanies: 0,
        activeInstitutions: 0,
      }),
    });
  });

  // Bloquer le reste des appels API non nécessaires pendant le pré-rendu.
  await page.route('**/api/**', async (route) => {
    const url = route.request().url();
    if (url.includes('platform-stats')) {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 204,
      body: '',
    });
  });
}

async function prerenderRoute(page, origin, route) {
  const url = `${origin}${route === '/' ? '/' : route}`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

  await page.waitForSelector('[data-seo-ready="true"]', {
    state: 'attached',
    timeout: 45000,
  });

  await page.waitForFunction(
    () => {
      const title = document.title;
      const canonical = document.querySelector('link[rel="canonical"]');
      const robots = document.querySelector('meta[name="robots"]');
      const h1 = document.querySelector('h1');
      return (
        Boolean(title) &&
        title !== 'Lirie' &&
        Boolean(canonical?.getAttribute('href')) &&
        Boolean(robots?.getAttribute('content')?.includes('index')) &&
        Boolean(h1?.textContent?.trim())
      );
    },
    { timeout: 45000 }
  );

  // Contrôle secondaire optionnel (timeout court).
  try {
    await page.waitForLoadState('networkidle', { timeout: 2000 });
  } catch (_) {
    /* ignore */
  }

  let html = await page.content();
  html = dedupeHeadTags(html);

  // Nettoyage session résiduelle éventuelle dans le DOM.
  html = html.replace(/activation_session_id/gi, '');

  const outFile = outputPathForRoute(route);
  // Pour `/`, on écrase build/index.html avec le HTML pré-rendu indexable.
  fs.writeFileSync(outFile, html, 'utf8');
  validatePrerenderedHtml(html, route);
  console.log(`[prerender] OK ${route} → ${path.relative(FRONTEND_ROOT, outFile)}`);
}

async function main() {
  if (!fs.existsSync(path.join(BUILD_DIR, 'index.html'))) {
    throw new Error('build/index.html introuvable. Exécutez d’abord npm run build:react.');
  }

  let server;
  let browser;
  try {
    server = await startStaticServer();
    try {
      browser = await chromium.launch({
        headless: true,
      });
    } catch (launchErr) {
      throw new Error(
        `Échec lancement Chromium Playwright: ${launchErr.message}. ` +
          'Exécutez « npx playwright install chromium » (déjà inclus dans npm run prerender).'
      );
    }
    const context = await browser.newContext({
      locale: 'fr-CH',
      timezoneId: 'Europe/Zurich',
      javaScriptEnabled: true,
      serviceWorkers: 'block',
    });
    const page = await context.newPage();
    await configurePage(page);

    for (const route of PUBLIC_ROUTES) {
      await prerenderRoute(page, server.origin, route);
    }

    await context.close();
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (server) {
      await server.close().catch(() => {});
    }
  }
}

const isDirectRun =
  process.argv[1] &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;

if (isDirectRun) {
  main().catch((err) => {
    console.error('[prerender] ÉCHEC', err);
    process.exit(1);
  });
}

export { PUBLIC_ROUTES, BUILD_DIR };
