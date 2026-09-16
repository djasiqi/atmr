/**
 * Une facture partenaire ne doit jamais passer par GET /invoices/{id}
 * ni par l’aperçu HTML client (catalogues isolés).
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';

const partnerInvoice = {
  id: 34,
  company_id: 1,
  status: 'draft',
  invoice_number: 'PARTNER-EM-2026-08-0097',
  is_partner_invoice: true,
  kind: 'partner',
  total_amount: 40.5,
  issued_at: '2026-09-07T00:00:00',
  due_date: '2026-09-17',
  period_year: 2026,
  period_month: 8,
  pdf_url: '/uploads/partner-invoices/0097.pdf',
  client: {
    institution_name: 'MT Genève',
    is_institution: true,
  },
  lines: [],
};

const mockGetInvoice = jest.fn(async () => {
  throw Object.assign(new Error('Facture non trouvée'), {
    response: { status: 404, data: { error: 'not_found' } },
  });
});

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: (...args) => mockGetInvoice(...args),
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
    forceRegenerateInvoicePdf: jest.fn(),
    regenerateInvoicePdf: jest.fn(),
    updateDraftInvoiceLine: jest.fn(),
  },
  formatCurrencyCHF: (n) => `${Number(n).toFixed(2)} CHF`,
}));

jest.mock('../../../../../utils/invoicePdfPrint', () => ({
  printPdfBytes: jest.fn(),
  preloadInvoicePdfPrint: () => Promise.resolve(),
}));

jest.mock('../../../../../utils/protectedPdf', () => ({
  downloadProtectedPdfAsFile: jest.fn(),
  fetchProtectedPdfBytes: jest.fn(),
  fetchProtectedPdfObjectUrl: jest.fn(async () => 'blob:partner-pdf'),
  openProtectedPdfInNewTab: jest.fn(),
}));

describe('DraftInvoiceEditorPanel — facture partenaire', () => {
  beforeEach(() => {
    mockGetInvoice.mockClear();
  });

  it('n’appelle pas GET /invoices/{id} et n’affiche pas l’erreur de chargement', async () => {
    render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={partnerInvoice}
      />
    );

    await waitFor(() => {
      expect(
        screen.queryByText('Impossible de charger la facture.')
      ).not.toBeInTheDocument();
    });
    expect(mockGetInvoice).not.toHaveBeenCalled();
    expect(
      screen.queryByText(/Aperçu HTML — le PDF officiel/i)
    ).not.toBeInTheDocument();
  });

  it('ignore un 404 /invoices/34 déjà parti quand le catalogue partenaire arrive', async () => {
    let rejectStandard;
    mockGetInvoice.mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          rejectStandard = reject;
        })
    );

    const { rerender } = render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={{ id: 34, company_id: 1, status: 'draft' }}
      />
    );

    await waitFor(() => {
      expect(mockGetInvoice).toHaveBeenCalledTimes(1);
    });

    rerender(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={partnerInvoice}
      />
    );

    rejectStandard(
      Object.assign(new Error('Facture non trouvée'), {
        response: { status: 404, data: { error: 'not_found' } },
      })
    );

    await waitFor(() => {
      expect(
        screen.queryByText('Impossible de charger la facture.')
      ).not.toBeInTheDocument();
    });
    expect(
      screen.queryByText(/Aperçu HTML — le PDF officiel/i)
    ).not.toBeInTheDocument();
  });
});
