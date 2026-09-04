import React, { useState } from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { invoiceService } from '../../../../../services/invoiceService';
import DisputeResolutionPanel from './DisputeResolutionPanel';

const dupont = {
  bookingId: 45705,
  patientName: 'Marie DUPONT',
  scheduledAt: '2026-08-16T09:00:00',
  amountHt: 40,
};

const klein = {
  bookingId: 45690,
  patientName: 'Arturo KLEIN',
  scheduledAt: '2026-08-02T08:00:00',
  amountHt: 40,
};

const disputePayload = {
  id: 1,
  booking_id: 45705,
  status: 'disputed',
  patient_name: 'Marie DUPONT',
  scheduled_at: '2026-08-16T09:00:00',
  amount_ht: 40,
  institution_reason_code: 'OTHER',
  institution_reason_text: 'OTHER: Pas de retour suite hospitalisation',
  evidence: [],
};

jest.mock('../../../../../services/invoiceService', () => ({
  invoiceService: {
    getBookingDispute: jest.fn(),
    respondBookingDispute: jest.fn(),
    addBookingDisputeEvidence: jest.fn(),
    submitBookingDispute: jest.fn(),
  },
  formatCurrencyCHF: (n) => `${Number(n).toFixed(2)} CHF`,
}));

const isInViewport = (el, viewport = { width: 1024, height: 768 }) => {
  const box = el.getBoundingClientRect();
  return (
    box.width > 0 &&
    box.height > 0 &&
    box.top < viewport.height &&
    box.bottom > 0 &&
    box.left < viewport.width &&
    box.right > 0
  );
};

const mockPanelBox = () => {
  const proto = Element.prototype;
  const original = proto.getBoundingClientRect;
  proto.getBoundingClientRect = function getBoundingClientRect() {
    if (this.getAttribute?.('data-testid') === 'dispute-resolution-panel') {
      return {
        top: 80,
        left: 200,
        bottom: 520,
        right: 720,
        width: 520,
        height: 440,
        x: 200,
        y: 80,
      };
    }
    return original.call(this);
  };
  return () => {
    proto.getBoundingClientRect = original;
  };
};

function InvoiceScrollShell({ onTreat }) {
  return (
    <div data-testid="invoice-scroll" style={{ height: 240, overflow: 'auto' }}>
      <div data-testid="invoice-top">Haut de la facture</div>
      <div style={{ height: 900 }}>Contenu long</div>
      <button
        type="button"
        data-testid="dispute-treat-45705"
        onClick={() => onTreat(dupont)}
      >
        Traiter la contestation
      </button>
    </div>
  );
}

function TreatHost({ firstRow = dupont }) {
  const [row, setRow] = useState(null);
  return (
    <>
      <InvoiceScrollShell onTreat={setRow} />
      <button
        type="button"
        data-testid="dispute-treat-45690"
        onClick={() => setRow(klein)}
      >
        Traiter Klein
      </button>
      {row ? (
        <DisputeResolutionPanel
          companyId={1}
          row={row}
          onClose={() => setRow(null)}
        />
      ) : null}
      <button type="button" data-testid="open-first" onClick={() => setRow(firstRow)}>
        Ouvrir
      </button>
    </>
  );
}

describe('DisputeResolutionPanel — overlay viewport', () => {
  let restoreBox;

  beforeEach(() => {
    restoreBox = mockPanelBox();
    invoiceService.getBookingDispute.mockImplementation(async (_companyId, bookingId) => ({
      data: {
        ...disputePayload,
        booking_id: bookingId,
        patient_name: bookingId === 45690 ? 'Arturo KLEIN' : 'Marie DUPONT',
      },
    }));
  });

  afterEach(() => {
    restoreBox();
  });

  it('apparaît tout de suite en overlay fixé, hors du scroll facture', async () => {
    const user = userEvent.setup();
    render(<TreatHost />);
    const scroller = screen.getByTestId('invoice-scroll');
    scroller.scrollTop = 0;
    await user.click(screen.getByTestId('dispute-treat-45705'));
    const overlay = screen.getByTestId('dispute-resolution-overlay');
    const panel = screen.getByTestId('dispute-resolution-panel');
    expect(overlay).toHaveAttribute('data-placement', 'viewport-fixed');
    expect(panel).toHaveAttribute('role', 'dialog');
    expect(overlay.contains(scroller)).toBe(false);
    expect(scroller.contains(panel)).toBe(false);
    expect(document.body.contains(overlay)).toBe(true);
    expect(isInViewport(panel)).toBe(true);
    expect(screen.getByText('Contestation — Marie DUPONT')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText('Autre')).toBeInTheDocument();
      expect(screen.getByText('Pas de retour suite hospitalisation')).toBeInTheDocument();
    });
    expect(screen.queryByText(/mauvais payeur \/ autre/i)).not.toBeInTheDocument();
  });

  it('reste visible après scroll tout en bas de la facture', async () => {
    const user = userEvent.setup();
    render(<TreatHost />);
    const scroller = screen.getByTestId('invoice-scroll');
    scroller.scrollTop = scroller.scrollHeight;
    await user.click(screen.getByTestId('dispute-treat-45705'));
    const panel = screen.getByTestId('dispute-resolution-panel');
    expect(scroller.contains(panel)).toBe(false);
    expect(isInViewport(panel)).toBe(true);
  });

  it('ferme uniquement la contestation (Escape / Fermer) puis se rouvre', async () => {
    const user = userEvent.setup();
    render(<TreatHost />);
    await user.click(screen.getByTestId('open-first'));
    expect(screen.getByTestId('dispute-resolution-panel')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByTestId('dispute-resolution-panel')).not.toBeInTheDocument();
    expect(screen.getByTestId('invoice-scroll')).toBeInTheDocument();
    await user.click(screen.getByTestId('open-first'));
    expect(screen.getByTestId('dispute-resolution-panel')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Fermer la contestation' }));
    expect(screen.queryByTestId('dispute-resolution-panel')).not.toBeInTheDocument();
  });

  it('enchaîne deux courses contestées sans rester sur l’ancienne', async () => {
    const user = userEvent.setup();
    render(<TreatHost />);
    await user.click(screen.getByTestId('open-first'));
    expect(await screen.findByText('Contestation — Marie DUPONT')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Fermer la contestation' }));
    await user.click(screen.getByTestId('dispute-treat-45690'));
    expect(await screen.findByText('Contestation — Arturo KLEIN')).toBeInTheDocument();
    expect(screen.queryByText('Contestation — Marie DUPONT')).not.toBeInTheDocument();
  });

  it('reste dans le viewport sur petit écran et n’est pas clipé par overflow', async () => {
    const user = userEvent.setup();
    const previous = { width: window.innerWidth, height: window.innerHeight };
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 375 });
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 667 });
    render(<TreatHost />);
    await user.click(screen.getByTestId('open-first'));
    const overlay = screen.getByTestId('dispute-resolution-overlay');
    const panel = screen.getByTestId('dispute-resolution-panel');
    expect(isInViewport(panel, { width: 375, height: 667 })).toBe(true);
    expect(window.getComputedStyle(overlay).overflow).not.toBe('visible');
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: previous.width });
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: previous.height });
  });

  it('affiche Continuer après le choix, sans quitter le dialogue', async () => {
    const user = userEvent.setup();
    render(<TreatHost />);
    await user.click(screen.getByTestId('open-first'));
    await waitFor(() => screen.getByLabelText("L'institution a raison"));
    await user.click(screen.getByLabelText("L'institution a raison"));
    await user.click(screen.getByTestId('dispute-continue'));
    expect(within(screen.getByTestId('dispute-resolution-panel')).getByText(/ne doit pas être facturée/i)).toBeInTheDocument();
  });
});
