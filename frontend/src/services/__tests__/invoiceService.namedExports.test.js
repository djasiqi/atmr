/**
 * Anti-régression : dispatchers du registre importés comme fonctions isolées.
 * Verrouille le lot « fuite de this » (paiement, envoyée, PDF, annulation).
 */
import apiClient from '../../utils/apiClient';
import {
  postPaymentForResource,
  markInvoiceAsSentForResource,
  forceRegenerateInvoicePdfForResource,
  cancelInvoiceResource,
  sendInvoiceByEmailForResource,
} from '../invoiceService';
import {
  resolveInvoiceResource,
  INVOICE_CATALOG,
} from '../../utils/invoiceCatalog';
import { getInvoiceCapabilities } from '../../utils/invoiceCapabilities';

jest.mock('../../utils/apiClient', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  },
}));

const COMPANY_ID = 1;

const STANDARD_0044 = {
  id: 44,
  invoice_number: 'EM-2026-08-0044',
  invoice_type: INVOICE_CATALOG.STANDARD,
  company_id: COMPANY_ID,
  balance_due: 90,
};

const PARTNER_INVOICE = {
  id: 34,
  invoice_number: 'PARTNER-EM-2026-08-0097',
  invoice_type: INVOICE_CATALOG.PARTNER,
  is_partner_invoice: true,
  company_id: COMPANY_ID,
};

/** Payload modal registre : 90.00 CHF, 20.09.2026, virement, sans référence. */
const PAYMENT_0044 = {
  amount: 90.0,
  paid_at: '2026-09-20',
  method: 'bank_transfer',
  reference: '',
};

describe('dispatchers isolés — STANDARD EM-2026-08-0044', () => {
  beforeEach(() => {
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    apiClient.put.mockReset();
    apiClient.delete.mockReset();
  });

  it('enregistre un paiement 90.00 CHF le 2026-09-20 par virement', async () => {
    apiClient.post.mockResolvedValue({ data: { id: 1, status: 'paid' } });

    await postPaymentForResource(STANDARD_0044, COMPANY_ID, PAYMENT_0044);

    expect(apiClient.post).toHaveBeenCalledTimes(1);
    const [url, payload] = apiClient.post.mock.calls[0];
    expect(url).toBe(
      `/invoices/companies/${COMPANY_ID}/invoices/${STANDARD_0044.id}/payments`
    );
    expect(url).not.toMatch(/partner-invoices/);
    expect(payload.amount).toBe(90.0);
    expect(payload.paid_at).toBe('2026-09-20');
    expect(payload.method).toBe('bank_transfer');
    expect(payload.reference === '' || payload.reference == null).toBe(true);
  });

  it('marque comme envoyée via le catalogue standard', async () => {
    apiClient.post.mockResolvedValue({ data: { status: 'sent' } });

    await markInvoiceAsSentForResource(STANDARD_0044, COMPANY_ID);

    expect(apiClient.post).toHaveBeenCalledWith(
      `/invoices/companies/${COMPANY_ID}/invoices/${STANDARD_0044.id}/send`,
      { send_method: 'paper' }
    );
    expect(apiClient.post.mock.calls[0][0]).not.toMatch(/partner-invoices/);
  });

  it('régénère le PDF via /invoices/{id}/regenerate-pdf', async () => {
    apiClient.post.mockResolvedValue({
      data: {
        pdf_url: '/uploads/invoices/em-2026-08-0044.pdf',
        pdf_generated_at: '2026-09-20T10:00:00Z',
      },
    });

    await forceRegenerateInvoicePdfForResource(STANDARD_0044, COMPANY_ID);

    expect(apiClient.post).toHaveBeenCalledWith(
      `/invoices/companies/${COMPANY_ID}/invoices/${STANDARD_0044.id}/regenerate-pdf`
    );
    expect(apiClient.post.mock.calls[0][0]).not.toMatch(/partner-invoices/);
  });

  it('annule la facture via le catalogue standard', async () => {
    apiClient.post.mockResolvedValue({ data: { status: 'cancelled' } });

    await cancelInvoiceResource(STANDARD_0044, COMPANY_ID);

    expect(apiClient.post).toHaveBeenCalledWith(
      `/invoices/companies/${COMPANY_ID}/invoices/${STANDARD_0044.id}/cancel`
    );
    expect(apiClient.post.mock.calls[0][0]).not.toMatch(/partner-invoices/);
  });
});

describe('dispatchers isolés — PARTNER (aucune fuite vers le catalogue standard)', () => {
  beforeEach(() => {
    apiClient.get.mockReset();
    apiClient.post.mockReset();
    apiClient.put.mockReset();
    apiClient.delete.mockReset();
  });

  it('route le paiement vers le catalogue partenaire', async () => {
    apiClient.post.mockResolvedValue({ data: { id: 9 } });

    await postPaymentForResource(PARTNER_INVOICE, COMPANY_ID, PAYMENT_0044);

    const [url, payload] = apiClient.post.mock.calls[0];
    expect(url).toBe(
      `/invoices/companies/${COMPANY_ID}/partner-invoices/${PARTNER_INVOICE.id}/payments`
    );
    expect(url).not.toMatch(/\/invoices\/34\//);
    expect(payload.amount).toBe(90.0);
    expect(payload.paid_at).toBe('2026-09-20');
    expect(payload.method).toBe('bank_transfer');
  });

  it('route le PDF partenaire vers /partner-invoices/{id}/pdf', () => {
    const resource = resolveInvoiceResource(PARTNER_INVOICE, COMPANY_ID);
    expect(resource.pdfApiUrl).toBe(
      `/invoices/companies/${COMPANY_ID}/partner-invoices/${PARTNER_INVOICE.id}/pdf`
    );
    expect(resource.pdfApiUrl).not.toMatch(/\/invoices\/34\/pdf/);
    expect(getInvoiceCapabilities(PARTNER_INVOICE, COMPANY_ID).canViewPdf).toBe(true);
  });

  it('régénère le PDF partenaire sans toucher /invoices/{id}', async () => {
    apiClient.post.mockResolvedValue({
      data: {
        pdf_url: '/uploads/partner-invoices/p-0097.pdf',
        pdf_generated_at: '2026-09-20T10:00:00Z',
      },
    });

    await forceRegenerateInvoicePdfForResource(PARTNER_INVOICE, COMPANY_ID);

    expect(apiClient.post).toHaveBeenCalledWith(
      `/invoices/companies/${COMPANY_ID}/partner-invoices/${PARTNER_INVOICE.id}/regenerate-pdf`
    );
    expect(apiClient.post.mock.calls[0][0]).not.toMatch(/\/invoices\/34\//);
  });

  it('n’envoie pas l’e-mail partenaire vers le catalogue standard', async () => {
    apiClient.post.mockResolvedValue({ data: { status: 'sent' } });

    await sendInvoiceByEmailForResource(PARTNER_INVOICE, COMPANY_ID, {
      recipient_email: 'partner@example.com',
    });

    expect(apiClient.post.mock.calls[0][0]).toMatch(
      /\/partner-invoices\/34\/send$/
    );
    expect(apiClient.post.mock.calls[0][0]).not.toMatch(/\/invoices\/34\//);
  });
});
