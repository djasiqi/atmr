import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import InvoiceSituation from './InvoiceSituation';

const exemple = {
  disponible: true,
  prestations_chf: '90.00',
  avoirs_chf: '-30.00',
  net_facture_chf: '60.00',
  acomptes_chf: '-20.00',
  paiements_complementaires_chf: '0.00',
  remboursements_confirmes_chf: '0.00',
  reste_a_payer_chf: '40.00',
  a_rembourser_chf: '0.00',
  events: [
    { at: '2026-03-01T10:00:00+00:00', label: 'Facture émise', reference: 'F-90', montant: '90.00' },
    { at: '2026-03-02T10:00:00+00:00', label: 'Acompte chauffeur déjà reçu', reference: 'ESP-20', montant: '-20.00' },
    { at: '2026-03-04T10:00:00+00:00', label: 'Avoir émis', reference: 'AV-1-2026-1', montant: '-30.00' },
  ],
  documents: [
    { label: 'Facture originale', reference: 'F-90', pdf_url: '/uploads/invoices/f-90.pdf', disponible: true },
    { label: 'Avoir', reference: 'AV-1-2026-1', pdf_url: '/uploads/credit_notes/av.pdf', disponible: true },
  ],
};

describe('situation affichée', () => {
  it('reprend les montants du serveur, y compris un net absent', () => {
    const { rerender } = render(<InvoiceSituation statement={exemple} />);
    expect(screen.getByText('Montant facturé').nextSibling).toHaveTextContent('90.00 CHF');
    expect(screen.getByText('Avoirs').nextSibling).toHaveTextContent('-30.00 CHF');
    expect(screen.getByText('Net facturé').nextSibling).toHaveTextContent('60.00 CHF');
    expect(screen.getByText('Acomptes reçus').nextSibling).toHaveTextContent('-20.00 CHF');
    expect(screen.getByText('Reste à payer').nextSibling).toHaveTextContent('40.00 CHF');
    expect(screen.queryByText('invoice_payments')).toBeNull();
    expect(screen.queryByText('credit_note')).toBeNull();
    expect(screen.queryByText('refundable_chf')).toBeNull();

    rerender(
      <InvoiceSituation
        statement={{ ...exemple, net_facture_chf: null, reste_a_payer_chf: null }}
      />
    );
    expect(screen.getByText('Net facturé').nextSibling).toHaveTextContent('Montant non indiqué');
    expect(screen.getByText('Reste à payer').nextSibling).toHaveTextContent('Montant non indiqué');
    expect(screen.queryByText('60.00 CHF')).toBeNull();
  });

  it('distingue un avoir d un paiement et un crédit remboursable d une dette', () => {
    render(
      <InvoiceSituation
        statement={{
          ...exemple,
          paiements_complementaires_chf: '-90.00',
          acomptes_chf: '0.00',
          reste_a_payer_chf: '0.00',
          a_rembourser_chf: '30.00',
          events: [
            { at: '2026-03-02T10:00:00+00:00', label: 'Paiement complémentaire', reference: 'VIR-90', montant: '-90.00' },
            { at: '2026-03-05T10:00:00+00:00', label: 'Avoir émis', reference: 'AV-1', montant: '-30.00' },
          ],
        }}
      />
    );
    expect(screen.getByText('Avoir émis')).toBeTruthy();
    expect(screen.getByText('Paiement complémentaire')).toBeTruthy();
    expect(screen.getByText('Reste à payer').nextSibling).toHaveTextContent('0.00 CHF');
    expect(screen.getByText('Montant à rembourser').nextSibling).toHaveTextContent('30.00 CHF');
  });

  it('conserve l ordre et les références reçus', () => {
    render(<InvoiceSituation statement={exemple} />);
    const text = document.body.textContent;
    const issued = text.indexOf('Facture émise');
    const deposit = text.indexOf('Acompte chauffeur déjà reçu');
    const credit = text.indexOf('Avoir émis');
    expect(issued).toBeLessThan(deposit);
    expect(deposit).toBeLessThan(credit);
    expect(text).toContain('ESP-20');
    expect(text).toContain('2026-03-02T10:00:00+00:00');
  });

  it('ouvre le PDF archivé et signale un document manquant', async () => {
    const onOpenPdf = jest.fn();
    const user = userEvent.setup();
    const { rerender } = render(
      <InvoiceSituation statement={exemple} onOpenPdf={onOpenPdf} />
    );
    await user.click(screen.getByRole('button', { name: /Consulter la facture F-90/ }));
    await user.click(screen.getByRole('button', { name: /Consulter l'avoir AV-1-2026-1/ }));
    expect(onOpenPdf).toHaveBeenNthCalledWith(1, '/uploads/invoices/f-90.pdf');
    expect(onOpenPdf).toHaveBeenNthCalledWith(2, '/uploads/credit_notes/av.pdf');
    expect(screen.queryByRole('button', { name: /régénér|regener|Émettre un avoir|Rembourser|Enregistrer paiement/i })).toBeNull();

    rerender(
      <InvoiceSituation
        statement={{
          ...exemple,
          documents: [
            { label: 'Facture originale', reference: 'F-90', pdf_url: null, disponible: false },
            { label: 'Avoir', reference: 'AV-1', pdf_url: null, disponible: false },
          ],
        }}
        onOpenPdf={onOpenPdf}
      />
    );
    expect(screen.getAllByText('Document indisponible')).toHaveLength(2);
    expect(screen.queryByRole('button', { name: /Consulter/ })).toBeNull();
  });

  it('affiche le refus d entreprise et le refus chauffeur', () => {
    const { rerender } = render(
      <InvoiceSituation error="Entreprise non trouvée ou accès refusé" />
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/refus/i);
    rerender(<InvoiceSituation error="Accès non autorisé" />);
    expect(screen.getByRole('alert')).toHaveTextContent('Accès non autorisé');
    expect(screen.queryByRole('button', { name: /Émettre un avoir|Rembourser|Enregistrer paiement/i })).toBeNull();
  });
});
