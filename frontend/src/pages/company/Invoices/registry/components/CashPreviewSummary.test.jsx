import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

import { presentCashPreview } from '../../../../../utils/invoiceLinesPreviewUi';
import CashPreviewSummary from './CashPreviewSummary';

const partial = presentCashPreview({
  estimated_subtotal_ht: '45.00',
  collected_chf: '20.00',
  remaining_due_chf: '25.00',
  preview_lines: [
    {
      booking_id: 70,
      amount_ht: 45,
      cash_legs: [
        {
          booking_id: 70,
          contractual_chf: '45.00',
          recognized_net_chf: '20.00',
          remaining_due_chf: '25.00',
        },
      ],
    },
  ],
});

describe('CashPreviewSummary', () => {
  it('affiche 45, 20 et 25 sans second bouton', () => {
    render(<CashPreviewSummary summary={partial} />);
    expect(screen.getByText('Montant des prestations')).toBeInTheDocument();
    expect(screen.getByText('Déjà encaissé par le chauffeur')).toBeInTheDocument();
    expect(screen.getByText('Reste à payer')).toBeInTheDocument();
    expect(screen.getByText('45.00 CHF')).toBeInTheDocument();
    expect(screen.getByText('20.00 CHF')).toBeInTheDocument();
    expect(screen.getByTestId('cash-preview-remaining')).toHaveTextContent('25.00 CHF');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('détaille l aller-retour 90 / 20 / 70', () => {
    const summary = presentCashPreview({
      estimated_subtotal_ht: '90.00',
      collected_chf: '20.00',
      remaining_due_chf: '70.00',
      preview_lines: [
        {
          booking_id: 71,
          amount_ht: 90,
          cash_legs: [
            {
              booking_id: 71,
              contractual_chf: '45.00',
              recognized_net_chf: '20.00',
              remaining_due_chf: '25.00',
            },
            {
              booking_id: 72,
              contractual_chf: '45.00',
              recognized_net_chf: '0.00',
              remaining_due_chf: '45.00',
            },
          ],
        },
      ],
    });
    render(<CashPreviewSummary summary={summary} />);
    expect(screen.getByText('90.00 CHF')).toBeInTheDocument();
    expect(screen.getByText('70.00 CHF')).toBeInTheDocument();
    expect(screen.getByTestId('cash-preview-leg-71')).toHaveTextContent('45.00 CHF');
    expect(screen.getByTestId('cash-preview-leg-71')).toHaveTextContent('25.00 CHF');
    expect(screen.getByTestId('cash-preview-leg-72')).toHaveTextContent('Reste 45.00 CHF');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('affiche un solde nul', () => {
    const summary = presentCashPreview({
      estimated_subtotal_ht: '45.00',
      collected_chf: '45.00',
      remaining_due_chf: '0.00',
      preview_lines: [
        {
          booking_id: 73,
          amount_ht: 45,
          cash_legs: [
            {
              booking_id: 73,
              contractual_chf: '45.00',
              recognized_net_chf: '45.00',
              remaining_due_chf: '0.00',
            },
          ],
        },
      ],
    });
    render(<CashPreviewSummary summary={summary} />);
    expect(screen.getByTestId('cash-preview-remaining')).toHaveTextContent('0.00 CHF');
    expect(screen.getAllByText('45.00 CHF').length).toBeGreaterThan(0);
  });

  it('n affiche rien si la prévisualisation ne porte pas d acompte', () => {
    const { container } = render(<CashPreviewSummary summary={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
