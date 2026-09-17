import { getInvoiceCapabilities } from '../invoiceCapabilities';

describe('getInvoiceCapabilities — catalogue partenaire', () => {
  const partnerDraft = {
    id: 34,
    company_id: 1,
    status: 'draft',
    invoice_number: 'PARTNER-EM-2026-08-0097',
    is_partner_invoice: true,
    amount_paid: 0,
    balance_due: 40.5,
  };

  test('un brouillon partenaire a les actions métier natives', () => {
    const caps = getInvoiceCapabilities(partnerDraft, 1);
    expect(caps.canEdit).toBe(true);
    expect(caps.canRegeneratePdf).toBe(true);
    expect(caps.canCancel).toBe(true);
    expect(caps.canSend).toBe(true);
    expect(caps.canAddPayment).toBe(true);
    expect(caps.canGenerateReminder).toBe(false);
    expect(caps.resource.regeneratePdfApiUrl).toContain('/partner-invoices/34/');
    expect(caps.resource.regeneratePdfApiUrl).not.toMatch('/invoices/34');
  });

  test('une facture partenaire payée n’est plus éditable', () => {
    const caps = getInvoiceCapabilities(
      { ...partnerDraft, status: 'paid', amount_paid: 40.5, balance_due: 0 },
      1
    );
    expect(caps.canEdit).toBe(false);
    expect(caps.canAddPayment).toBe(false);
    expect(caps.canRegeneratePdf).toBe(false);
    expect(caps.canCancel).toBe(false);
  });
});
