/**
 * Patient, clinique et partenaire ouvrent le même shell d'édition.
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import InvoiceDraftEditModal from './InvoiceDraftEditModal';

const mockPatient = {
  id: 11,
  company_id: 1,
  status: 'draft',
  invoice_number: 'EM-PAT-11',
  billing_strategy: 's1_patient',
  total_amount: 80,
  subtotal_amount: 80,
  issued_at: '2026-08-01',
  due_date: '2026-08-15',
  period_year: 2026,
  period_month: 8,
  client: { first_name: 'Ada', last_name: 'Lovelace' },
  lines: [
    {
      id: 1,
      type: 'ride',
      description: 'Trajet domicile',
      line_total: 80,
      service_date: '2026-08-02',
    },
  ],
};

const mockClinic = {
  ...mockPatient,
  id: 22,
  invoice_number: 'EM-CLI-22',
  billing_strategy: 's2_clinic_monthly',
  client: { institution_name: 'Clinique du Lac', is_institution: true },
};

const mockPartner = {
  id: 34,
  company_id: 1,
  status: 'draft',
  invoice_number: 'PARTNER-EM-34',
  is_partner_invoice: true,
  billing_strategy: 'partner_monthly',
  editor_header: true,
  subject_contact: 'Facturation',
  total_amount: 40,
  subtotal_amount: 40,
  issued_at: '2026-08-01',
  due_date: '2026-08-20',
  period_year: 2026,
  period_month: 8,
  recipient_name: 'MobilCar',
  recipient_contact: 'Facturation',
  client: { institution_name: 'MobilCar', is_institution: true },
  lines: [
    {
      id: 1,
      description: 'Transfert',
      amount: 40,
      quantity: 1,
      unit_price: 40,
      service_date: '2026-08-03',
      source_type: 'booking_transfer',
    },
  ],
};

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: jest.fn(async (_companyId, invoiceId) =>
    invoiceId === 22 ? mockClinic : mockPatient
  ),
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
    getPartnerInvoice: jest.fn(async () => mockPartner),
    updatePartnerInvoice: jest.fn(),
    forceRegenerateInvoicePdf: jest.fn(async () => ({ pdf_url: '/pdf/std.pdf' })),
    forceRegeneratePartnerInvoicePdf: jest.fn(async () => ({
      pdf_url: '/pdf/partner.pdf',
    })),
    updateDraftInvoiceLine: jest.fn(),
    addDraftCustomLine: jest.fn(),
    removeDraftInvoiceLine: jest.fn(),
    applyDraftGlobalDiscount: jest.fn(),
    applyDraftPerLineDiscounts: jest.fn(),
    removeDraftGlobalDiscount: jest.fn(),
    sendInvoiceByEmail: jest.fn(),
    sendPartnerInvoiceByEmail: jest.fn(),
    markInvoiceAsSent: jest.fn(),
    markPartnerInvoiceAsSent: jest.fn(),
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

async function expectSharedToolbar(invoiceNumber) {
  expect(await screen.findByTestId('invoice-draft-toolbar')).toBeInTheDocument();
  expect(screen.getByText('Aperçu facture')).toBeInTheDocument();
  expect(screen.getAllByText(invoiceNumber).length).toBeGreaterThan(0);
  expect(screen.getByRole('button', { name: 'Remises' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Ajouter une ligne supplémentaire HT' })).toBeEnabled();
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
  expect(screen.getByRole('document', { name: 'Aperçu facture' })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Envoyer' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Marquer envoyée' })).toBeEnabled();
}

describe('éditeur de facture unique', () => {
  it('patient, clinique et partenaire partagent la barre et l’aperçu', async () => {
    const { unmount } = render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={mockPatient}
        onOpenSendEmail={() => {}}
        onMarkAsSent={() => {}}
      />
    );
    await expectSharedToolbar('EM-PAT-11');
    expect(screen.getByText('Client / Patient')).toBeInTheDocument();
    unmount();

    const clinicView = render(
      <DraftInvoiceEditorPanel
        open
        companyId={1}
        initialInvoice={mockClinic}
        onOpenSendEmail={() => {}}
        onMarkAsSent={() => {}}
      />
    );
    await expectSharedToolbar('EM-CLI-22');
    expect(screen.getByText('Contact clinique')).toBeInTheDocument();
    clinicView.unmount();

    render(
      <InvoiceDraftEditModal
        open
        companyId={1}
        initialInvoice={mockPartner}
        onClose={() => {}}
        onOpenSendEmail={() => {}}
        onMarkAsSent={() => {}}
      />
    );
    await expectSharedToolbar('PARTNER-EM-34');
    expect(screen.getByText('Partenaire / Contact')).toBeInTheDocument();
    expect(screen.getByText('Facturation')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Informations facture' })).toBeEnabled();
    expect(screen.queryByText('Éditer le brouillon partenaire')).not.toBeInTheDocument();
  });
});
