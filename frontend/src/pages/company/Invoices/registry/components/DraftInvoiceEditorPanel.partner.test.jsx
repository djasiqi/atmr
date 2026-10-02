/**
 * Facture partenaire : même éditeur, jamais GET /invoices/{id}.
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
  billing_strategy: 'partner_monthly',
  subject_contact: 'Accueil',
  editor_header: true,
  total_amount: 40.5,
  subtotal_amount: 40.5,
  vat_amount: 0,
  issued_at: '2026-09-07T00:00:00',
  due_date: '2026-09-17',
  period_year: 2026,
  period_month: 8,
  recipient_name: 'MT Genève',
  recipient_contact: 'Accueil',
  pdf_url: '/uploads/partner-invoices/0097.pdf',
  client: {
    institution_name: 'MT Genève',
    is_institution: true,
  },
  lines: [
    {
      id: 1,
      type: 'ride',
      description: 'Course Genève',
      amount: 40.5,
      line_total: 40.5,
      quantity: 1,
      unit_price: 40.5,
      service_date: '2026-08-04',
    },
  ],
};

const mockGetInvoice = jest.fn(async () => {
  throw Object.assign(new Error('Facture non trouvée'), {
    response: { status: 404, data: { error: 'not_found' } },
  });
});

const mockGetPartnerInvoice = jest.fn(async () => partnerInvoice);

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: (...args) => mockGetInvoice(...args),
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
    getPartnerInvoice: (...args) => mockGetPartnerInvoice(...args),
    updatePartnerInvoice: jest.fn(),
    forceRegenerateInvoicePdf: jest.fn(),
    forceRegeneratePartnerInvoicePdf: jest.fn(async () => ({
      pdf_url: '/uploads/partner-invoices/0097.pdf',
    })),
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
    mockGetPartnerInvoice.mockClear();
  });

  it('charge le détail partenaire et affiche le même aperçu', async () => {
    render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={partnerInvoice}
      />
    );

    expect(await screen.findByTestId('invoice-draft-toolbar')).toBeInTheDocument();
    expect(screen.getByText('Aperçu facture')).toBeInTheDocument();
    expect(screen.getByRole('document', { name: 'Aperçu facture' })).toBeInTheDocument();
    expect(screen.getAllByText('PARTNER-EM-2026-08-0097').length).toBeGreaterThan(0);
    expect(screen.getByText('Partenaire / Contact')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Informations facture' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Remises' })).toBeEnabled();
    expect(mockGetInvoice).not.toHaveBeenCalled();
    expect(mockGetPartnerInvoice).toHaveBeenCalledWith(1, 34, { cacheBust: true });
    expect(screen.queryByText('Impossible de charger la facture.')).not.toBeInTheDocument();
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
      expect(mockGetPartnerInvoice).toHaveBeenCalled();
    });
    expect(screen.queryByText('Impossible de charger la facture.')).not.toBeInTheDocument();
    expect(screen.getByRole('document', { name: 'Aperçu facture' })).toBeInTheDocument();
  });
});
