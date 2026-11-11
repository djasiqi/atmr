import React, { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import DraftInvoiceEditorPanel from './DraftInvoiceEditorPanel';
import InvoiceLivePreview from './InvoiceLivePreview';
import { buildPreparationInvoice } from '../../../../../utils/invoicePreparationDraft';
import { getInvoice } from '../../../../../services/invoiceService';

jest.mock('../../../../../services/invoiceService', () => ({
  getInvoice: jest.fn(),
  formatCurrencyCHF: (value) => `${Number(value).toFixed(2)} CHF`,
  invoiceService: {
    fetchBillingSettings: jest.fn(async () => ({ vat_applicable: false })),
    getPartnerInvoice: jest.fn(),
  },
}));

const preparation = buildPreparationInvoice({
  payerType: 'clinic',
  periodYear: 2026,
  periodMonth: 10,
  recipientName: 'Clinique du Lac',
  previewLines: [
    {
      booking_id: 3,
      scheduled_at: '2026-10-06T09:10:00',
      patient_name: 'Claire Bonvin',
      description: 'Anieres → clinique',
      amount_ht: 72,
    },
  ],
});

test('le mode préparation affiche l’aperçu live sans numéro ni outils PDF', () => {
  render(
    <DraftInvoiceEditorPanel
      open
      mode="preparation"
      initialInvoice={preparation}
      companyId={1}
    />
  );
  expect(screen.getByText('Aperçu facture')).toBeInTheDocument();
  expect(screen.getByText('Brouillon de préparation')).toBeInTheDocument();
  expect(screen.queryByText('Facture non créée')).not.toBeInTheDocument();
  expect(screen.queryByText('Facture')).not.toBeInTheDocument();
  expect(screen.getByText('Client : BONVIN Claire')).toBeInTheDocument();
  expect(screen.getByText('Trajet : Anieres → clinique')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Régénérer le PDF/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /télécharger le PDF/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Imprimer le PDF/i })).not.toBeInTheDocument();
  expect(getInvoice).not.toHaveBeenCalled();
});

test('le même aperçu affiche le numéro seulement une fois la facture persistée', () => {
  function Harness() {
    const [persisted, setPersisted] = useState(false);
    const invoice = persisted
      ? {
          ...preparation,
          preparation: false,
          id: 44,
          invoice_number: 'EM-2026-10-0008',
          status: 'draft',
        }
      : preparation;
    return (
      <>
        <button type="button" onClick={() => setPersisted(true)}>
          Basculer
        </button>
        <InvoiceLivePreview invoice={invoice} companyVatApplicable={false} />
      </>
    );
  }
  render(<Harness />);
  expect(screen.queryByText('Facture non créée')).not.toBeInTheDocument();
  expect(screen.queryByText('EM-2026-10-0008')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Basculer' }));
  expect(screen.getByText('EM-2026-10-0008')).toBeInTheDocument();
});
