import {
  presentPartnerInvoiceSummary,
  presentPatientInvoiceSummary,
} from '../payerInvoiceSummaryUi';

describe('payerInvoiceSummaryUi', () => {
  it('résumé Patient = uniquement ce que ce patient doit payer', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Charlotte CAVADINI',
      transports_count: 1,
      unbilled_total_amount: 40,
      can_generate: true,
    });
    expect(summary).toMatchObject({
      visible: true,
      hasBillable: true,
      displayName: 'Charlotte CAVADINI',
      transportsCount: 1,
      totalHt: 40,
      blocked: false,
    });
  });

  it('patient à compléter : visible, non facturable', () => {
    const summary = presentPatientInvoiceSummary({
      display_name: 'Alice MARTIN',
      segments_count: 2,
      unbilled_total_amount: 80,
      can_generate: false,
    });
    expect(summary.hasBillable).toBe(false);
    expect(summary.blocked).toBe(true);
    expect(summary.transportsCount).toBe(2);
  });

  it('résumé Partenaire = uniquement les transferts validés de ce partenaire', () => {
    const summary = presentPartnerInvoiceSummary({
      partner_company_name: 'Partenaire Test',
      validated_unbilled_transfers_count: 4,
      unbilled_transfers_count: 6,
      estimated_subtotal_ht: 160,
    });
    expect(summary).toMatchObject({
      visible: true,
      hasBillable: true,
      transportsCount: 4,
      totalHt: 160,
    });
    expect(summary.excluded).toMatchObject({ visible: true, count: 2 });
  });
});
