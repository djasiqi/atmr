import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';

const draftInvoice = {
  id: 88,
  company_id: 1,
  status: 'draft',
  invoice_number: 'EM-2026-08-0002',
  total_ht: 280,
  issued_at: '2026-09-04T10:00:00',
  due_date: '2026-10-04',
  period_year: 2026,
  period_month: 8,
  clinic_name: "Clinique les Hauts d'Anières",
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

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: jest.fn(async () => draftInvoice),
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
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

describe('DraftInvoiceEditorPanel — toolbar après Prepare', () => {
  it('UI-DRAFT-8 à 12 : barre historique câblée + InvoiceLivePreview', async () => {
    render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={draftInvoice}
        toolbarSubtitle={"Août 2026 · Clinique les Hauts d'Anières"}
      />
    );

    const toolbar = await screen.findByTestId('invoice-draft-toolbar');
    expect(toolbar).toHaveAttribute('aria-label', 'Brouillon et aperçu facture');
    expect(screen.getByText('Aperçu facture')).toBeInTheDocument();
    expect(screen.getByText("Août 2026 · Clinique les Hauts d'Anières")).toBeInTheDocument();

    expect(screen.getByRole('button', { name: 'Remises' })).toBeEnabled();
    expect(
      screen.getByRole('button', { name: 'Ajouter une ligne supplémentaire HT' })
    ).toBeEnabled();
    expect(
      screen.getByRole('button', { name: 'Ouvrir l’édition des lignes sous l’aperçu PDF' })
    ).toBeEnabled();
    await waitFor(() => {
      expect(
        screen.getByRole('button', {
          name: 'Régénérer le PDF et actualiser les données depuis le serveur',
        })
      ).toBeEnabled();
    });
    expect(screen.getByRole('button', { name: 'Agrandir la zone d’aperçu PDF' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Plein écran dans le navigateur' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Générer et télécharger le PDF' })).toBeEnabled();
    expect(
      screen.getByRole('button', { name: 'Imprimer le PDF de la facture sans quitter l’aperçu' })
    ).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Ouvrir le PDF dans un nouvel onglet' })).toBeNull();

    expect(screen.getByRole('document', { name: 'Aperçu facture' })).toBeInTheDocument();
  });
});
