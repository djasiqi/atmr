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

  test('une facture partenaire envoyée n’est plus éditable ni régénérable', () => {
    const caps = getInvoiceCapabilities({ ...partnerDraft, status: 'sent' }, 1);
    expect(caps.canEdit).toBe(false);
    expect(caps.canRegeneratePdf).toBe(false);
    expect(caps.canAddPayment).toBe(true);
    expect(caps.canViewPdf).toBe(true);
  });

  test('une facture patient envoyée est en lecture seule', () => {
    const caps = getInvoiceCapabilities(
      {
        id: 12,
        company_id: 1,
        status: 'sent',
        invoice_number: 'EM-12',
        amount_paid: 0,
        balance_due: 40,
        pdf_url: '/uploads/invoices/em-12.pdf',
      },
      1
    );
    expect(caps.canEdit).toBe(false);
    expect(caps.canRegeneratePdf).toBe(false);
    expect(caps.canAddPayment).toBe(true);
    expect(caps.canViewPdf).toBe(true);
    expect(caps.canDuplicate).toBe(false);
    expect(caps.canCorrect).toBe(true);
    expect(caps.canOpenEditor).toBe(true);
    expect(caps.primaryActionLabel).toBe('Corriger la facture');
  });

  test('un brouillon patient s’édite ; pas de correction émise', () => {
    const caps = getInvoiceCapabilities(
      {
        id: 99,
        company_id: 1,
        status: 'draft',
        invoice_number: null,
        amount_paid: 0,
        balance_due: 48.45,
      },
      1
    );
    expect(caps.canEdit).toBe(true);
    expect(caps.canOpenEditor).toBe(true);
    expect(caps.canCorrect).toBe(false);
    expect(caps.canDuplicate).toBe(false);
    expect(caps.primaryActionLabel).toBe('Éditer la facture');
  });

  test('une facture partiellement payée ouvre l’éditeur en mode Corriger', () => {
    const caps = getInvoiceCapabilities(
      {
        id: 3170,
        company_id: 73011,
        status: 'partially_paid',
        invoice_number: 'EM-2026-11-0004',
        amount_paid: 20,
        balance_due: 28.45,
      },
      73011
    );
    expect(caps.canEdit).toBe(false);
    expect(caps.canOpenEditor).toBe(true);
    expect(caps.canDuplicate).toBe(false);
    expect(caps.canCorrect).toBe(true);
    expect(caps.primaryActionLabel).toBe('Corriger la facture');
  });

  test('un statut inconnu ou vide n’ouvre pas l’édition', () => {
    const unknown = getInvoiceCapabilities(
      { id: 1, company_id: 1, status: 'UNPAID', amount_paid: 0 },
      1
    );
    const missing = getInvoiceCapabilities({ id: 2, company_id: 1, amount_paid: 0 }, 1);
    expect(unknown.canEdit).toBe(false);
    expect(unknown.canRegeneratePdf).toBe(false);
    expect(missing.canEdit).toBe(false);
    expect(missing.canRegeneratePdf).toBe(false);
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
