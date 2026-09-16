/**
 * Anti-régression permanente — contrat figé « Régénérer PDF » (CLOSED).
 * SAVE si nécessaire → forceRegenerateInvoicePdf. Voir docs/facturation/regenerer-pdf-contrat.md.
 */
import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';

const mockInvoice = {
  id: 88,
  company_id: 1,
  status: 'draft',
  invoice_number: 'EM-2026-08-0002',
  total_ht: 280,
  issued_at: '2026-09-04T10:00:00',
  due_date: '2026-10-04',
  period_year: 2026,
  period_month: 8,
  pdf_url: '/uploads/invoices/old.pdf',
  lines: [
    {
      id: 1,
      type: 'ride',
      description: 'Trajet Chemin des Courbes 9 → HUG',
      line_total: 40,
      service_date: '2026-08-02',
      reservation_id: 45697,
    },
  ],
};

const mockForceRegenerateInvoicePdf = jest.fn(async () => ({
  pdf_url: '/uploads/invoices/new.pdf',
  pdf_generated_at: '2026-09-16T10:00:00Z',
}));

const mockUpdateDraftInvoiceLine = jest.fn(async (_c, _i, lineId, body) => ({
  invoice: {
    ...mockInvoice,
    lines: mockInvoice.lines.map((ln) =>
      ln.id === lineId ? { ...ln, ...body } : ln
    ),
  },
}));

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: jest.fn(async () => ({
    id: 88,
    company_id: 1,
    status: 'draft',
    invoice_number: 'EM-2026-08-0002',
    total_ht: 280,
    issued_at: '2026-09-04T10:00:00',
    due_date: '2026-10-04',
    period_year: 2026,
    period_month: 8,
    pdf_url: '/uploads/invoices/old.pdf',
    lines: [
      {
        id: 1,
        type: 'ride',
        description: 'Trajet Chemin des Courbes 9 → HUG',
        line_total: 40,
        service_date: '2026-08-02',
        reservation_id: 45697,
      },
    ],
  })),
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
    forceRegenerateInvoicePdf: (...args) => mockForceRegenerateInvoicePdf(...args),
    regenerateInvoicePdf: (...args) => mockForceRegenerateInvoicePdf(...args),
    updateDraftInvoiceLine: (...args) => mockUpdateDraftInvoiceLine(...args),
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
  fetchProtectedPdfObjectUrl: jest.fn(),
  openProtectedPdfInNewTab: jest.fn(),
}));

describe('DraftInvoiceEditorPanel — régénération forcée (contrat figé)', () => {
  beforeEach(() => {
    mockForceRegenerateInvoicePdf.mockClear();
    mockUpdateDraftInvoiceLine.mockClear();
  });

  it('le bouton icône appelle forceRegenerateInvoicePdf puis recharge', async () => {
    render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={mockInvoice}
        toolbarSubtitle={"Août 2026 · Clinique les Hauts d'Anières"}
      />
    );

    const btn = await screen.findByRole('button', {
      name: 'Régénérer le PDF et actualiser les données depuis le serveur',
    });
    await waitFor(() => {
      expect(btn).toBeEnabled();
    });
    fireEvent.click(btn);

    await waitFor(() => {
      expect(mockForceRegenerateInvoicePdf).toHaveBeenCalledTimes(1);
    });
    expect(mockForceRegenerateInvoicePdf).toHaveBeenCalledWith(1, 88);
  });

  it('sauvegarde une ligne sale avant de régénérer', async () => {
    render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={mockInvoice}
        toolbarSubtitle={"Août 2026 · Clinique les Hauts d'Anières"}
      />
    );

    fireEvent.click(
      await screen.findByRole('button', {
        name: 'Ouvrir l’édition des lignes sous l’aperçu PDF',
      })
    );

    const desc = await screen.findByLabelText(/Libellé/i);
    fireEvent.change(desc, { target: { value: 'Trajet mis à jour' } });

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Régénérer le PDF et actualiser les données depuis le serveur',
      })
    );

    await waitFor(() => {
      expect(mockUpdateDraftInvoiceLine).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(mockForceRegenerateInvoicePdf).toHaveBeenCalledWith(1, 88);
    });
    const saveOrder = mockUpdateDraftInvoiceLine.mock.invocationCallOrder[0];
    const regenOrder = mockForceRegenerateInvoicePdf.mock.invocationCallOrder[0];
    expect(saveOrder).toBeLessThan(regenOrder);
  });
});
