import {
  buildInvoiceLineFilterHaystack,
  filterInvoiceLines,
  invoiceLineMatchesFilter,
  normalizeInvoiceLineSearchText,
  patientNameToSearchVariants,
  serviceDateIsoToSearchVariants,
} from '../invoiceLineFilter';

describe('invoiceLineFilter', () => {
  const sampleLine = {
    id: 42,
    type: 'ride',
    description:
      "Trajet Chemin des Courbes 9, 1247 Anières → Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205, Genève",
    line_total: 80,
    reservation_id: 9001,
    adjustment_note: '',
    line_meta: {
      patient_name: 'BOUCHARDY Jean-michel',
      service_date: '2026-05-02',
    },
  };

  it('normalise accents et casse', () => {
    expect(normalizeInvoiceLineSearchText('Élisabeth')).toBe('elisabeth');
  });

  it('génère des variantes de date', () => {
    const v = serviceDateIsoToSearchVariants('2026-05-02');
    expect(v).toContain('02.05.2026');
    expect(v).toContain('02/05/2026');
    expect(v).toContain('mai');
    expect(v).toContain('mai 2026');
    expect(v).toContain('05.2026');
  });

  it('génère des variantes nom prénom / prénom nom', () => {
    const v = patientNameToSearchVariants('BOUCHARDY Jean-michel');
    expect(v.some((x) => normalizeInvoiceLineSearchText(x).includes('bouchardy'))).toBe(true);
    expect(v.some((x) => x.includes('Jean-michel BOUCHARDY'))).toBe(true);
  });

  it('inclut client et date dans le haystack', () => {
    const h = buildInvoiceLineFilterHaystack(sampleLine);
    expect(h).toContain('bouchardy');
    expect(h).toContain('jean-michel');
    expect(h).toContain('02.05.2026');
    expect(h).toContain('hug');
  });

  it('filtre par nom prénom', () => {
    expect(invoiceLineMatchesFilter(sampleLine, 'Jean-michel Bouchardy')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, 'BOUCHARDY')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, 'Dupont')).toBe(false);
  });

  it('filtre par date partielle ou complète', () => {
    expect(invoiceLineMatchesFilter(sampleLine, '02.05.2026')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, '02.05')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, 'mai 2026')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, '03.05.2026')).toBe(false);
  });

  it('filtre combiné nom + date', () => {
    expect(invoiceLineMatchesFilter(sampleLine, 'Bouchardy 02.05')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, 'Bouchardy 03.05')).toBe(false);
  });

  it('filtre par libellé, n° ligne et réservation', () => {
    expect(invoiceLineMatchesFilter(sampleLine, 'HUG')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, '42')).toBe(true);
    expect(invoiceLineMatchesFilter(sampleLine, '9001')).toBe(true);
  });

  it('filterInvoiceLines conserve l’ordre', () => {
    const lines = [
      sampleLine,
      { ...sampleLine, id: 43, line_meta: { patient_name: 'AELLEN Jules', service_date: '2026-05-04' } },
    ];
    const out = filterInvoiceLines(lines, 'Aellen');
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe(43);
  });
});
