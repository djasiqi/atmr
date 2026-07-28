/**
 * Validation des HTML pré-rendus (unicité balises + motifs sensibles).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(FRONTEND_ROOT, 'build');

const EMAIL_ALLOWLIST = new Set([
  'info@lirie.ch',
  'privacy@lirie.ch',
]);

const PLACEHOLDER_EMAIL_RE =
  /^(nom|exemple|example|contact|user|test|marie)@/i;

const FORBIDDEN_PATTERNS = [
  { name: 'activation_session_id', re: /activation_session_id/i },
  { name: 'access_token', re: /access_token/i },
  { name: 'refresh_token', re: /refresh_token/i },
  { name: 'patient_id', re: /patient_id/i },
  { name: 'booking_id', re: /booking_id/i },
  // Coordonnées JSON / attributs (évite faux positifs textuels trop larges)
  { name: 'latitude_json', re: /"latitude"\s*:/i },
  { name: 'longitude_json', re: /"longitude"\s*:/i },
];

const ROUTES = [
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

function countMatches(html, re) {
  return (html.match(re) || []).length;
}

function assertOne(html, label, re) {
  const n = countMatches(html, re);
  if (n !== 1) {
    throw new Error(`[seo-validate] ${label}: attendu 1, trouvé ${n}`);
  }
}

function scanSensitive(html, route) {
  for (const { name, re } of FORBIDDEN_PATTERNS) {
    if (re.test(html)) {
      throw new Error(
        `[seo-validate] Contenu sensible détecté (${name}) sur ${route}`
      );
    }
  }

  const emails = html.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|ch|fr)\b/gi) || [];
  for (const email of emails) {
    const lower = email.toLowerCase();
    if (EMAIL_ALLOWLIST.has(lower)) continue;
    if (PLACEHOLDER_EMAIL_RE.test(lower)) continue;
    // Placeholders formulaires type organisation.ch / exemple.ch
    if (/@(organisation|exemple|example|test)\./i.test(lower)) continue;
    throw new Error(
      `[seo-validate] Email non autorisé dans HTML pré-rendu (${email}) sur ${route}`
    );
  }

  if (/0\s+transports?\s+coordonn/i.test(html) || />\s*0\s*<\/span>\s*<span[^>]*>entreprises partenaires/i.test(html)) {
    throw new Error(
      `[seo-validate] Compteurs artificiels à zéro détectés sur ${route}`
    );
  }
}

/**
 * @param {string} html
 * @param {string} route
 */
export function validatePrerenderedHtml(html, route) {
  if (!html.includes('data-seo-ready="true"')) {
    throw new Error(`[seo-validate] data-seo-ready manquant sur ${route}`);
  }

  assertOne(html, `${route} title`, /<title[^>]*>[\s\S]*?<\/title>/gi);
  assertOne(html, `${route} description`, /<meta[^>]*name=["']description["'][^>]*>/gi);
  assertOne(html, `${route} canonical`, /<link[^>]*rel=["']canonical["'][^>]*>/gi);
  assertOne(html, `${route} robots`, /<meta[^>]*name=["']robots["'][^>]*>/gi);
  assertOne(
    html,
    `${route} json-ld`,
    /<script[^>]*type=["']application\/ld\+json["'][^>]*>[\s\S]*?<\/script>/gi
  );

  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = (titleMatch?.[1] || '').trim();
  if (!title || title === 'Lirie') {
    throw new Error(`[seo-validate] Title invalide sur ${route}: ${title}`);
  }

  const robotsMatch = html.match(
    /<meta[^>]*name=["']robots["'][^>]*content=["']([^"']+)["'][^>]*>/i
  ) || html.match(
    /<meta[^>]*content=["']([^"']+)["'][^>]*name=["']robots["'][^>]*>/i
  );
  const robots = robotsMatch?.[1] || '';
  if (!/index/i.test(robots) || /noindex/i.test(robots)) {
    throw new Error(`[seo-validate] robots doit être index,follow sur ${route}: ${robots}`);
  }

  const canonicalMatch = html.match(
    /<link[^>]*rel=["']canonical["'][^>]*href=["']([^"']+)["'][^>]*>/i
  ) || html.match(
    /<link[^>]*href=["']([^"']+)["'][^>]*rel=["']canonical["'][^>]*>/i
  );
  const canonical = canonicalMatch?.[1] || '';
  if (!canonical.startsWith('https://www.lirie.ch')) {
    throw new Error(`[seo-validate] Canonical invalide sur ${route}: ${canonical}`);
  }
  if (route !== '/' && canonical.endsWith('/')) {
    throw new Error(`[seo-validate] Canonical avec slash final sur ${route}`);
  }

  if (!/<h1[\s>]/i.test(html)) {
    throw new Error(`[seo-validate] H1 manquant sur ${route}`);
  }

  const ldMatch = html.match(
    /<script[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/i
  );
  try {
    const parsed = JSON.parse(ldMatch[1]);
    if (!parsed || !parsed['@graph']) {
      throw new Error('JSON-LD sans @graph');
    }
    const blob = JSON.stringify(parsed).toLowerCase();
    if (blob.includes('est une entreprise de transport') || blob.includes('"@type":"taxi"')) {
      throw new Error('JSON-LD présente LIRIE comme transporteur');
    }
  } catch (err) {
    throw new Error(`[seo-validate] JSON-LD invalide sur ${route}: ${err.message}`);
  }

  scanSensitive(html, route);
}

function fileForRoute(route) {
  if (route === '/') return path.join(BUILD_DIR, 'index.html');
  return path.join(BUILD_DIR, route.replace(/^\//, ''), 'index.html');
}

export function validateAllPrerenderedFiles() {
  for (const route of ROUTES) {
    const file = fileForRoute(route);
    if (!fs.existsSync(file)) {
      throw new Error(`[seo-validate] Fichier manquant: ${file}`);
    }
    const html = fs.readFileSync(file, 'utf8');
    validatePrerenderedHtml(html, route);
    console.log(`[seo-validate] OK ${route}`);
  }
}

const isDirectRun =
  process.argv[1] &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;

if (isDirectRun) {
  try {
    validateAllPrerenderedFiles();
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
}
