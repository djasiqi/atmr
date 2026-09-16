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
});
