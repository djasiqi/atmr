#!/usr/bin/env node
/**
 * Smoke tests SEO-01C pour une URL de preview ou de production.
 *
 * Usage:
 *   node scripts/seo-preview-smoke.mjs https://xxx.vercel.app
 *   node scripts/seo-preview-smoke.mjs https://www.lirie.ch
 */
const base = (process.argv[2] || '').replace(/\/$/, '');

if (!base) {
  console.error('Usage: node scripts/seo-preview-smoke.mjs <preview-or-prod-url>');
  process.exit(1);
}

async function fetchText(path) {
  const url = `${base}${path}`;
  const res = await fetch(url, { redirect: 'follow' });
  const text = await res.text();
  return { url, status: res.status, headers: res.headers, text };
}

async function fetchHead(path) {
  const url = `${base}${path}`;
  const res = await fetch(url, { method: 'HEAD', redirect: 'manual' });
  return {
    url,
    status: res.status,
    location: res.headers.get('location'),
    contentType: res.headers.get('content-type'),
  };
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function main() {
  const errors = [];

  try {
    const pro = await fetchText('/professionnel');
    assert(pro.status === 200, `/professionnel status ${pro.status}`);
    assert(
      pro.text.includes('rel="canonical" href="https://www.lirie.ch/professionnel"') ||
        pro.text.includes("rel='canonical' href='https://www.lirie.ch/professionnel'"),
      'canonical professionnel manquante'
    );
    assert(
      /Gestion des transports pour EMS, cliniques et institutions/i.test(pro.text),
      'titre métier professionnel absent du HTML'
    );
    assert(/index,\s*follow/i.test(pro.text), 'index,follow absent sur /professionnel');

    const unknown = await fetchText('/route-privee-inconnue');
    assert(unknown.status === 200, `route inconnue status ${unknown.status}`);
    assert(pro.text !== unknown.text, 'HTML professionnel identique au shell (pré-rendu non servi)');
    assert(/noindex/i.test(unknown.text), 'noindex absent sur route inconnue');

    for (const privatePath of ['/login', '/activate-account', '/dashboard/company/test']) {
      const page = await fetchText(privatePath);
      assert(page.status === 200 || page.status === 401 || page.status === 403, `${privatePath} status ${page.status}`);
      if (page.status === 200) {
        assert(/noindex/i.test(page.text), `noindex absent sur ${privatePath}`);
        assert(!/content=["']index,\s*follow/i.test(page.text), `index,follow indésirable sur ${privatePath}`);
      }
    }

    const sitemap = await fetchText('/sitemap.xml');
    assert(sitemap.status === 200, 'sitemap.xml non accessible');
    assert((sitemap.text.match(/<loc>/g) || []).length === 10, 'sitemap doit contenir 10 URLs');
    assert(!/login|dashboard|activate-account/i.test(sitemap.text), 'sitemap contient des URLs privées');

    const robots = await fetchText('/robots.txt');
    assert(robots.status === 200, 'robots.txt non accessible');
    assert(/Sitemap:\s*https:\/\/www\.lirie\.ch\/sitemap\.xml/i.test(robots.text), 'Sitemap non déclaré');
    assert(/Disallow:\s*\/dashboard\//i.test(robots.text), 'Disallow dashboard manquant');

    const slash = await fetchHead('/professionnel/');
    // Suivi manuel : 308/301 vers sans slash, ou déjà sans slash selon CDN.
    if ([301, 302, 307, 308].includes(slash.status)) {
      assert(
        /\/professionnel\/?$/.test(slash.location || '') &&
          !(slash.location || '').endsWith('professionnel/'),
        `trailing slash redirect inattendu: ${slash.location}`
      );
    }

    console.log(`[seo-smoke] OK sur ${base}`);
  } catch (err) {
    errors.push(err.message);
  }

  // Apex (uniquement si on teste www — documenté séparément)
  if (base.includes('www.lirie.ch')) {
    try {
      const res = await fetch('https://lirie.ch/professionnel', { method: 'HEAD', redirect: 'manual' });
      const loc = res.headers.get('location') || '';
      if (![301, 302, 307, 308].includes(res.status) || !loc.includes('www.lirie.ch/professionnel')) {
        errors.push(
          `Redirection apex→www attendue (status=${res.status}, location=${loc}). Configurer le domaine primaire dans Vercel.`
        );
      } else {
        console.log('[seo-smoke] OK apex → www');
      }
    } catch (err) {
      errors.push(`Test apex: ${err.message}`);
    }
  }

  if (errors.length) {
    console.error('[seo-smoke] ÉCHECS:');
    for (const e of errors) console.error(` - ${e}`);
    process.exit(1);
  }
}

main();
