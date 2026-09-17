import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import PartnerInvoiceDraftEditModal from './PartnerInvoiceDraftEditModal';

const mockGetPartnerInvoice = jest.fn();
const mockUpdatePartnerInvoice = jest.fn();
const mockForceRegeneratePartnerInvoicePdf = jest.fn();

jest.mock('../../../../../services/invoiceService', () => ({
  getPartnerInvoice: (...args) => mockGetPartnerInvoice(...args),
  updatePartnerInvoice: (...args) => mockUpdatePartnerInvoice(...args),
  forceRegeneratePartnerInvoicePdf: (...args) =>
    mockForceRegeneratePartnerInvoicePdf(...args),
}));

describe('PartnerInvoiceDraftEditModal', () => {
  const invoice = {
    id: 34,
    company_id: 1,
    invoice_number: 'PARTNER-EM-2026-08-0097',
    is_partner_invoice: true,
    status: 'draft',
    notes: '',
    recipient_name: 'MT Genève',
    recipient_address: '',
    recipient_contact: '',
    period_year: 2026,
    period_month: 8,
    issued_at: '2026-08-01',
    due_date: '2026-09-17',
    total_amount: 40.5,
    lines: [
      {
        id: 1,
        description: 'Course initiale',
        quantity: 1,
        unit_price: 40.5,
        amount: 40.5,
      },
    ],
  };

  beforeEach(() => {
    mockGetPartnerInvoice.mockReset();
    mockUpdatePartnerInvoice.mockReset();
    mockForceRegeneratePartnerInvoicePdf.mockReset();
    mockGetPartnerInvoice.mockResolvedValue(invoice);
    mockUpdatePartnerInvoice.mockResolvedValue({
      ...invoice,
      lines: [
        {
          id: 1,
          description: 'Course corrigée',
          quantity: 1,
          unit_price: 45.5,
          amount: 45.5,
        },
      ],
      total_amount: 45.5,
    });
  });

  it('charge et enregistre via les endpoints partenaires uniquement', async () => {
    render(
      <PartnerInvoiceDraftEditModal
        open
        companyId={1}
        initialInvoice={invoice}
        onClose={() => {}}
      />
    );

    await screen.findByDisplayValue('Course initiale');
    expect(mockGetPartnerInvoice).toHaveBeenCalledWith(1, 34, { cacheBust: true });

    fireEvent.change(screen.getByDisplayValue('Course initiale'), {
      target: { value: 'Course corrigée' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Enregistrer' }));

    await waitFor(() => {
      expect(mockUpdatePartnerInvoice).toHaveBeenCalledTimes(1);
    });
    const [, , payload] = mockUpdatePartnerInvoice.mock.calls[0];
    expect(payload.lines[0].description).toBe('Course corrigée');
    expect(mockUpdatePartnerInvoice.mock.calls[0][0]).toBe(1);
    expect(mockUpdatePartnerInvoice.mock.calls[0][1]).toBe(34);
  });
});
