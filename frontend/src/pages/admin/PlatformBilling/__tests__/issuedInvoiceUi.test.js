import {
  displayInvoiceLineLabel,
  FALLBACK_SUPPORT_HOURLY_RATE,
  formatSupportHoursLabel,
  isSupportInvoiceLine,
  SUPPORT_LINE_TYPE,
  syncDerivedInvoiceLine,
} from '../issuedInvoiceUi';

describe('lignes support facture plateforme', () => {
  it('détecte une ligne qté × prix sans libellé comme support', () => {
    expect(
      isSupportInvoiceLine({
        calculation_mode: 'UNIT_PRICE',
        label: '',
        line_type: 'ADJUSTMENT',
        quantity: '49',
        unit_amount: '45',
      })
    ).toBe(true);
  });

  it('génère le libellé heures × tarif', () => {
    expect(formatSupportHoursLabel('49', '45')).toBe(
      'Support plateforme — 49 h à 45 CHF/h'
    );
    expect(FALLBACK_SUPPORT_HOURLY_RATE).toBe('45');
  });

  it('pose support_time et le libellé sur une ligne vide en qté × prix', () => {
    const next = syncDerivedInvoiceLine({
      calculation_mode: 'UNIT_PRICE',
      label: '',
      line_type: 'ADJUSTMENT',
      quantity: '49',
      unit_amount: '45',
    });
    expect(next.line_type).toBe(SUPPORT_LINE_TYPE);
    expect(next.label).toBe('Support plateforme — 49 h à 45 CHF/h');
  });

  it('n’écrase pas un libellé custom hors support', () => {
    const line = {
      calculation_mode: 'UNIT_PRICE',
      label: 'Forfait licences',
      line_type: 'ADJUSTMENT',
      quantity: '2',
      unit_amount: '50',
    };
    expect(isSupportInvoiceLine(line)).toBe(false);
    expect(syncDerivedInvoiceLine(line)).toEqual(line);
    expect(displayInvoiceLineLabel(line)).toBe('Forfait licences');
  });

  it('affiche le libellé support même si le champ description est vide', () => {
    expect(
      displayInvoiceLineLabel({
        calculation_mode: 'UNIT_PRICE',
        label: '',
        line_type: 'ADJUSTMENT',
        quantity: '49',
        unit_amount: '45',
      })
    ).toBe('Support plateforme — 49 h à 45 CHF/h');
  });
});
