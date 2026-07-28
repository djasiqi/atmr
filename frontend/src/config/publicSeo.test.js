import { listPublicSeoEntries, PUBLIC_SEO_PATHS, SEO_BASE_URL } from './publicSeo';

describe('publicSeo config', () => {
  it('expose exactement 10 routes publiques', () => {
    expect(PUBLIC_SEO_PATHS).toHaveLength(10);
    expect(listPublicSeoEntries()).toHaveLength(10);
  });

  it('garantit des titres et descriptions uniques', () => {
    const entries = listPublicSeoEntries();
    const titles = entries.map((e) => e.title);
    const descriptions = entries.map((e) => e.description);
    expect(new Set(titles).size).toBe(titles.length);
    expect(new Set(descriptions).size).toBe(descriptions.length);
  });

  it('utilise des canonicals absolues www sans slash final (sauf racine)', () => {
    for (const entry of listPublicSeoEntries()) {
      expect(entry.canonicalUrl.startsWith(SEO_BASE_URL)).toBe(true);
      expect(entry.canonicalUrl.includes('://lirie.ch')).toBe(false);
      if (entry.path === '/') {
        expect(entry.canonicalUrl).toBe(`${SEO_BASE_URL}/`);
      } else {
        expect(entry.canonicalUrl.endsWith('/')).toBe(false);
        expect(entry.canonicalUrl).toBe(`${SEO_BASE_URL}${entry.path}`);
      }
    }
  });

  it('n’inclut aucune donnée patient ou trajet dans les métadonnées', () => {
    const blob = JSON.stringify(listPublicSeoEntries());
    expect(blob).not.toMatch(/patient_id/i);
    expect(blob).not.toMatch(/booking_id/i);
    expect(blob).not.toMatch(/access_token/i);
    expect(blob).not.toMatch(/latitude/i);
    expect(blob).not.toMatch(/longitude/i);
  });

  it('présente LIRIE comme plateforme de coordination, pas transporteur', () => {
    for (const entry of listPublicSeoEntries()) {
      expect(entry.title).toMatch(/LIRIE/i);
      expect(entry.description.toLowerCase()).not.toMatch(
        /lirie est (une|un) (entreprise de )?transport/
      );
      const json = JSON.stringify(entry.structuredData);
      expect(json).toMatch(/Organization/);
      expect(json).toMatch(/WebSite/);
      expect(json).toMatch(/WebPage/);
      expect(json).toMatch(/n’exécute pas elle-même|coordination/i);
    }
  });

  it('limite SoftwareApplication aux pages prévues', () => {
    const withSoftware = listPublicSeoEntries().filter((e) =>
      JSON.stringify(e.structuredData).includes('SoftwareApplication')
    );
    const paths = withSoftware.map((e) => e.path).sort();
    expect(paths).toEqual(['/', '/conduire', '/professionnel']);
  });

  it('ajoute FAQPage uniquement sur /aide', () => {
    const aide = listPublicSeoEntries().find((e) => e.path === '/aide');
    expect(JSON.stringify(aide.structuredData)).toMatch(/FAQPage/);
    const others = listPublicSeoEntries().filter((e) => e.path !== '/aide');
    for (const entry of others) {
      expect(JSON.stringify(entry.structuredData)).not.toMatch(/FAQPage/);
    }
  });
});
