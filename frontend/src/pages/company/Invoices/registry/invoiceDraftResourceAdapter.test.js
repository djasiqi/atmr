/**
 * Isolation des catalogues : un id partagé ne doit jamais traverser l'autre endpoint.
 */
import { createInvoiceDraftAdapter } from './invoiceDraftResourceAdapter';

const mockGetInvoice = jest.fn();
const mockInvoiceService = {
  getPartnerInvoice: jest.fn(),
  updatePartnerInvoice: jest.fn(),
  updateDraftInvoiceLine: jest.fn(),
  addDraftCustomLine: jest.fn(),
  removeDraftInvoiceLine: jest.fn(),
  applyDraftGlobalDiscount: jest.fn(),
  applyDraftPerLineDiscounts: jest.fn(),
  removeDraftGlobalDiscount: jest.fn(),
  forceRegenerateInvoicePdf: jest.fn(),
  forceRegeneratePartnerInvoicePdf: jest.fn(),
  sendInvoiceByEmail: jest.fn(),
  sendPartnerInvoiceByEmail: jest.fn(),
  markInvoiceAsSent: jest.fn(),
  markPartnerInvoiceAsSent: jest.fn(),
};

jest.mock('../../../../services/invoiceService', () => ({
  getInvoice: (...args) => mockGetInvoice(...args),
  invoiceService: {
    getPartnerInvoice: (...args) => mockInvoiceService.getPartnerInvoice(...args),
    updatePartnerInvoice: (...args) => mockInvoiceService.updatePartnerInvoice(...args),
    updateDraftInvoiceLine: (...args) => mockInvoiceService.updateDraftInvoiceLine(...args),
    addDraftCustomLine: (...args) => mockInvoiceService.addDraftCustomLine(...args),
    removeDraftInvoiceLine: (...args) => mockInvoiceService.removeDraftInvoiceLine(...args),
    applyDraftGlobalDiscount: (...args) => mockInvoiceService.applyDraftGlobalDiscount(...args),
    applyDraftPerLineDiscounts: (...args) =>
      mockInvoiceService.applyDraftPerLineDiscounts(...args),
    removeDraftGlobalDiscount: (...args) =>
      mockInvoiceService.removeDraftGlobalDiscount(...args),
    forceRegenerateInvoicePdf: (...args) =>
      mockInvoiceService.forceRegenerateInvoicePdf(...args),
    forceRegeneratePartnerInvoicePdf: (...args) =>
      mockInvoiceService.forceRegeneratePartnerInvoicePdf(...args),
    sendInvoiceByEmail: (...args) => mockInvoiceService.sendInvoiceByEmail(...args),
    sendPartnerInvoiceByEmail: (...args) =>
      mockInvoiceService.sendPartnerInvoiceByEmail(...args),
    markInvoiceAsSent: (...args) => mockInvoiceService.markInvoiceAsSent(...args),
    markPartnerInvoiceAsSent: (...args) =>
      mockInvoiceService.markPartnerInvoiceAsSent(...args),
  },
}));

const sharedId = 34;

describe('invoiceDraftResourceAdapter', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetInvoice.mockResolvedValue({
      id: sharedId,
      invoice_number: 'EM-34',
      lines: [],
      total_amount: 10,
    });
    mockInvoiceService.getPartnerInvoice.mockResolvedValue({
      id: sharedId,
      invoice_number: 'PARTNER-EM-34',
      is_partner_invoice: true,
      lines: [{ id: 1, description: 'Course', amount: 40, quantity: 1, unit_price: 40 }],
      total_amount: 40,
      subtotal_amount: 40,
      vat_amount: 0,
      recipient_name: 'MT Genève',
      recipient_contact: 'Desk',
    });
    mockInvoiceService.updatePartnerInvoice.mockResolvedValue({
      id: sharedId,
      invoice_number: 'PARTNER-EM-34',
      is_partner_invoice: true,
      lines: [],
      total_amount: 36,
    });
    mockInvoiceService.forceRegeneratePartnerInvoicePdf.mockResolvedValue({
      pdf_url: '/uploads/partner-invoices/34.pdf',
    });
    mockInvoiceService.forceRegenerateInvoicePdf.mockResolvedValue({
      pdf_url: '/uploads/invoices/34.pdf',
    });
  });

  it('isole le chargement standard et partenaire pour le même id', async () => {
    const standard = createInvoiceDraftAdapter({ id: sharedId, invoice_number: 'EM-34' }, 7);
    const partner = createInvoiceDraftAdapter(
      { id: sharedId, invoice_number: 'PARTNER-EM-34', is_partner_invoice: true },
      7
    );

    const standardInvoice = await standard.load();
    const partnerInvoice = await partner.load();

    expect(mockGetInvoice).toHaveBeenCalledWith(7, sharedId, { cacheBust: true });
    expect(mockInvoiceService.getPartnerInvoice).toHaveBeenCalledWith(7, sharedId, {
      cacheBust: true,
    });
    expect(standardInvoice.catalog_type).toBe('standard');
    expect(partnerInvoice.catalog_type).toBe('partner');
    expect(partnerInvoice.billing_strategy).toBe('partner_monthly');
    expect(partnerInvoice.lines[0].type).toBe('ride');
    expect(partnerInvoice.lines[0].line_total).toBe(40);
    expect(partnerInvoice.subject_contact).toBe('Desk');
    expect(standard.pdfApiUrl({ pdf_url: '/uploads/invoices/34.pdf' })).toBe(
      '/invoices/companies/7/invoices/34/pdf'
    );
    expect(partner.pdfApiUrl(partnerInvoice)).toBe(
      '/invoices/companies/7/partner-invoices/34/pdf'
    );
  });

  it('route les mutations partenaire uniquement vers updatePartnerInvoice', async () => {
    const partner = createInvoiceDraftAdapter(
      { id: sharedId, is_partner_invoice: true, invoice_number: 'PARTNER-1' },
      7
    );
    await partner.updateLine(9, { description: 'Course', line_total: 12, adjustment_note: null });
    await partner.addCustomLine({ description: 'Attente', line_total: 5, qty: 1 });
    await partner.applyGlobalDiscount({ global_discount_percent: 10 });
    await partner.removeDiscount();
    await partner.regeneratePdf();
    await partner.markAsSent();

    expect(mockInvoiceService.updateDraftInvoiceLine).not.toHaveBeenCalled();
    expect(mockInvoiceService.forceRegenerateInvoicePdf).not.toHaveBeenCalled();
    expect(mockInvoiceService.markInvoiceAsSent).not.toHaveBeenCalled();
    expect(mockInvoiceService.updatePartnerInvoice).toHaveBeenCalled();
    expect(mockInvoiceService.forceRegeneratePartnerInvoicePdf).toHaveBeenCalledWith(7, sharedId);
    expect(mockInvoiceService.markPartnerInvoiceAsSent).toHaveBeenCalledWith(7, sharedId);
    const commands = mockInvoiceService.updatePartnerInvoice.mock.calls.map((call) => call[2].command);
    expect(commands).toEqual([
      undefined,
      'add_custom_line',
      'apply_global_discount',
      'remove_discount',
    ]);
  });

  it('route les mutations standard sans endpoint partenaire', async () => {
    mockInvoiceService.updateDraftInvoiceLine.mockResolvedValue({
      invoice: { id: sharedId, lines: [], invoice_number: 'EM-34' },
    });
    const standard = createInvoiceDraftAdapter({ id: sharedId, invoice_number: 'EM-34' }, 7);
    await standard.updateLine(3, { description: 'Trajet', line_total: 20 });
    await standard.regeneratePdf();
    expect(mockInvoiceService.updatePartnerInvoice).not.toHaveBeenCalled();
    expect(mockInvoiceService.forceRegeneratePartnerInvoicePdf).not.toHaveBeenCalled();
    expect(mockInvoiceService.updateDraftInvoiceLine).toHaveBeenCalledWith(
      7,
      sharedId,
      3,
      { description: 'Trajet', line_total: 20 }
    );
    expect(mockInvoiceService.forceRegenerateInvoicePdf).toHaveBeenCalledWith(7, sharedId);
  });
});
