# Indexation SEO LIRIE (www.lirie.ch)

> **Périmètre verrouillé :** améliorer l’indexabilité des dix pages publiques existantes sans modifier l’architecture des espaces applicatifs ni introduire de nouvelles pages métier. Toute extension du sitemap, du JSON-LD ou du périmètre de pré-rendu nécessite une validation séparée.

## Décision

- **GO implémentation** (livré dans le dépôt).
- **GO fusion** si SEO-01A et SEO-01B sont verts (tests config + `build/`).
- **GO production** uniquement après smoke sur une **preview Vercel**.

Source de vérité : artefacts dans `frontend/build/`, puis HTML réellement servi par Vercel.

## Lots

| Lot | Contenu | Déploiement |
|-----|---------|-------------|
| SEO-01A | sitemap, robots, `PublicSeo`, 10 routes, OG/JSON-LD, `trailingSlash: false`, éditorial | Autonome (sans `noindex` global seul) |
| SEO-01B | `noindex` fail-closed + pré-rendu Playwright + tests `build/` | **Atomique** avec le shell `index.html` |
| SEO-01C | Domaines Vercel, smoke preview, headers | Preview → prod |
| SEO-02 | Contenu / FAQ / `llms.txt` (pages métier nouvelles = validation séparée) | Après indexabilité |
| SEO-03 | Search Console, Bing, IndexNow, analytics SEO | Après indexabilité |

## Fichiers clés

- [`frontend/public/robots.txt`](../../frontend/public/robots.txt)
- [`frontend/public/sitemap.xml`](../../frontend/public/sitemap.xml) — 10 URLs
- [`frontend/public/llms.txt`](../../frontend/public/llms.txt)
- [`frontend/public/index.html`](../../frontend/public/index.html) — `noindex` par défaut
- [`frontend/src/config/publicSeo.js`](../../frontend/src/config/publicSeo.js)
- [`frontend/src/components/seo/`](../../frontend/src/components/seo/)
- [`frontend/scripts/prerender-public-pages.mjs`](../../frontend/scripts/prerender-public-pages.mjs)
- [`frontend/scripts/validate-prerendered-html.mjs`](../../frontend/scripts/validate-prerendered-html.mjs)
- [`frontend/scripts/seo-preview-smoke.mjs`](../../frontend/scripts/seo-preview-smoke.mjs)
- [`frontend/vercel.json`](../../frontend/vercel.json) — `trailingSlash: false`, **pas** de `cleanUrls`, pas de catch-all SPA

## Commandes locales

```bash
cd frontend
npm test -- --watchAll=false --testPathPattern=publicSeo.test.js
npm run build                 # build:react + prerender
npm run test:seo-build        # valide build/<route>/index.html
npm run test:seo-smoke -- https://<preview>.vercel.app
```

Sur **Vercel**, le pré-rendu utilise `@sparticuz/chromium` + `playwright-core` (libs système Playwright classiques absentes). En local, Playwright Chromium standard est installé automatiquement si besoin.

## SEO-01C — checklist preview (obligatoire avant prod)

```bash
# Pré-rendu servi (≠ shell)
curl -fsSL https://<preview>/professionnel \
  | grep -F 'rel="canonical" href="https://www.lirie.ch/professionnel"'

curl -fsSL https://<preview>/professionnel > /tmp/professionnel.html
curl -fsSL https://<preview>/route-privee-inconnue > /tmp/shell.html
cmp /tmp/professionnel.html /tmp/shell.html   # doit différer

# Routes privées connues
for p in /login /activate-account /dashboard/company/test; do
  curl -fsSL "https://<preview>$p" | grep -F 'noindex'
done

# Découverte
curl -fsSL https://<preview>/sitemap.xml
curl -fsSL https://<preview>/robots.txt
curl -fsSL https://<preview>/llms.txt

# Ou script
npm run test:seo-smoke -- https://<preview>
```

### Domaine canonique (dashboard Vercel)

1. Primary domain : `www.lirie.ch`
2. `lirie.ch` → redirection permanente vers `https://www.lirie.ch` (conserver path + query)
3. Vérifier : `curl -I https://lirie.ch/professionnel` → 301/308 vers `https://www.lirie.ch/professionnel`

Ne pas ajouter de fallback SPA catch-all ni `cleanUrls: true`.

## SEO-03 — autorité externe (manuel)

1. **Google Search Console** — propriété `https://www.lirie.ch/` ; soumettre `https://www.lirie.ch/sitemap.xml`
2. **Bing Webmaster Tools** — idem ; activer IndexNow si disponible
3. Cohérence NAP : LIRIE, `info@lirie.ch`, `+41 22 552 03 02`, Genève / Suisse romande
4. Analytics : brancher `window.__LIRIE_SEO_ANALYTICS__ = (event, props) => { ... }` (voir [`seoAnalytics.js`](../../frontend/src/utils/seoAnalytics.js)). Aucune donnée patient / trajet / token.

## Pages éditoriales (hors périmètre actuel)

Validation séparée requise avant ajout au sitemap / pré-rendu :

- `/coordination-transport-etablissement-sante`
- `/transport-medical-non-urgent`
- `/transport-pmr-geneve` (uniquement si partenaires Genève réellement actifs)

## Matrice GO (rappel)

- [ ] SEO-01A + 01B verts en local (`build/` validé)
- [ ] Preview : pré-rendu servi, privés en `noindex`
- [ ] Apex → www ; trailing slash géré
- [ ] Sitemap = 10 URLs ; pas de stats 0 trompeuses
- [ ] Une seule balise title / description / canonical / robots / JSON-LD par page publique
