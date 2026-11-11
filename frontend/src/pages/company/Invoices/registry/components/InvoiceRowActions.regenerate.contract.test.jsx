/**
 * Anti-régression permanente — contrat figé « Régénérer PDF » (CLOSED).
 * InvoiceRowActions ne doit que déléguer : aucune logique de génération ici.
 * Voir docs/facturation/regenerer-pdf-contrat.md.
 */
import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import InvoiceRowActions from './InvoiceRowActions';
import { invoiceService } from '../../../../../services/invoiceService';

jest.mock('../../../../../utils/protectedPdf', () => ({
  openProtectedPdfInNewTab: jest.fn(),
}));

describe('InvoiceRowActions — délégation régénération (contrat figé)', () => {
  it('le menu appelle uniquement onRegeneratePdf, jamais le service', () => {
    const onRegeneratePdf = jest.fn();
    const forceSpy = jest.spyOn(invoiceService, 'forceRegenerateInvoicePdf');

    render(
      <InvoiceRowActions
        invoice={{
          id: 99,
          status: 'draft',
          pdf_url: '/uploads/invoices/old.pdf',
          reminder_level: 0,
        }}
        onRegeneratePdf={onRegeneratePdf}
      />
    );

    fireEvent.click(screen.getByTitle('Actions'));
    fireEvent.click(screen.getByRole('menuitem', { name: /Regenerer PDF/i }));

    expect(onRegeneratePdf).toHaveBeenCalledTimes(1);
    expect(forceSpy).not.toHaveBeenCalled();
    forceSpy.mockRestore();
  });

  it('facture émise : Corriger via l’éditeur unique, jamais Créer un correctif', () => {
    const onEditDraft = jest.fn();
    render(
      <InvoiceRowActions
        invoice={{
          id: 3170,
          company_id: 73011,
          status: 'partially_paid',
          invoice_number: 'EM-2026-11-0004',
          pdf_url: '/uploads/invoices/em.pdf',
          reminder_level: 0,
          amount_paid: 20,
          balance_due: 28.45,
        }}
        onEditDraft={onEditDraft}
        onDuplicate={jest.fn()}
      />
    );

    fireEvent.click(screen.getByTitle('Actions'));
    expect(screen.getByRole('menuitem', { name: /Corriger la facture/i })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: /^Éditer la facture$/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: /Creer un correctif/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: /^Voir la facture$/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('menuitem', { name: /Corriger la facture/i }));
    expect(onEditDraft).toHaveBeenCalled();
  });
});

