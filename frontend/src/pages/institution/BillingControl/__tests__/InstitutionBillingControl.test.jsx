import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import InstitutionBillingControl from '../InstitutionBillingControl';

const mockRefetch = jest.fn();

jest.mock('../../../../hooks/useInstitutionData', () => ({
  useInstitutionMe: jest.fn(),
  useInstitutionPatients: jest.fn(),
  useBillingControlBookings: jest.fn(),
  useValidateBillingControlBooking: jest.fn(),
  useMarkBillingControlAnomaly: jest.fn(),
  useReopenBillingControlBooking: jest.fn(),
  useChangeBillingControlPayer: jest.fn(),
}));

jest.mock('sonner', () => ({
  toast: { success: jest.fn(), error: jest.fn() },
}));

const hooks = require('../../../../hooks/useInstitutionData');

const listPayload = {
  items: [
    {
      booking_id: 101,
      scheduled_time: '2026-09-02T10:00:00',
      patient: { display_name: 'Mme X', institution_patient_id: 5 },
      segment_type: 'outbound',
      pickup: 'Domicile',
      dropoff: 'Clinique',
      transport_company: { company_id: 7, display_name: 'Emmenez-moi' },
      payer: { type: 'clinic', display_name: 'Clinique' },
      control: { effective_status: 'pending_review' },
      billing: { editable: true, locked: false, invoiced: false },
    },
    {
      booking_id: 102,
      scheduled_time: '2026-09-02T15:00:00',
      patient: { display_name: 'Mme X', institution_patient_id: 5 },
      segment_type: 'return',
      pickup: 'Clinique',
      dropoff: 'Domicile',
      transport_company: { company_id: 7, display_name: 'Emmenez-moi' },
      payer: { type: 'patient', display_name: 'Patient' },
      control: {
        effective_status: 'validated',
        validated_by_display_name: 'Marc',
        validated_at: '2026-09-01T15:42:00',
      },
      billing: { editable: true, locked: false, invoiced: false },
    },
  ],
  summary: {
    total: 2,
    payer_clinic: 1,
    payer_patient: 1,
    validated: 1,
    pending_review: 1,
    anomaly: 0,
  },
  pagination: { page: 1, page_size: 50, total: 2, total_pages: 1 },
};

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <InstitutionBillingControl />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('InstitutionBillingControl — U05–U16', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    hooks.useInstitutionMe.mockReturnValue({
      data: { institution_role: 'institution_admin' },
    });
    hooks.useInstitutionPatients.mockReturnValue({ data: { patients: [] } });
    hooks.useBillingControlBookings.mockReturnValue({
      data: listPayload,
      isLoading: false,
      isError: false,
      error: null,
      refetch: mockRefetch,
      isFetching: false,
    });
    hooks.useValidateBillingControlBooking.mockReturnValue({
      mutateAsync: jest.fn().mockResolvedValue({ success: true }),
    });
    hooks.useMarkBillingControlAnomaly.mockReturnValue({
      mutateAsync: jest.fn().mockResolvedValue({ success: true }),
    });
    hooks.useReopenBillingControlBooking.mockReturnValue({
      mutateAsync: jest.fn().mockResolvedValue({ success: true }),
    });
    hooks.useChangeBillingControlPayer.mockReturnValue({
      mutateAsync: jest.fn().mockResolvedValue({ success: true }),
    });
  });

  it('U05 — liste + summary chargés', () => {
    renderPage();
    expect(screen.getByTestId('billing-control-summary')).toHaveTextContent('2');
    expect(screen.getByTestId('billing-control-summary')).toHaveTextContent('Validés');
    expect(screen.getByTestId('billing-control-table')).toBeInTheDocument();
  });

  it('U06 — filtres présents', () => {
    renderPage();
    expect(screen.getByLabelText('Période')).toBeInTheDocument();
    expect(screen.getByLabelText('Statut')).toBeInTheDocument();
    expect(screen.getByLabelText('Filtre payeur')).toBeInTheDocument();
    expect(screen.getByLabelText('Transporteur')).toBeInTheDocument();
    expect(screen.getByLabelText('Patient')).toBeInTheDocument();
  });

  it('U07 — pagination affichée quand plusieurs pages', () => {
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        ...listPayload,
        pagination: { page: 1, page_size: 1, total: 2, total_pages: 2 },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: mockRefetch,
      isFetching: false,
    });
    renderPage();
    expect(screen.getByTestId('billing-control-pagination')).toBeInTheDocument();
  });

  it('U08 — regroupement visuel A/R', () => {
    renderPage();
    expect(screen.getAllByTestId('billing-control-group')).toHaveLength(1);
    expect(screen.getByText('Aller')).toBeInTheDocument();
    expect(screen.getByText('Retour')).toBeInTheDocument();
  });

  it('U09/U10 — changement payeur déclenche mutation serveur', async () => {
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockResolvedValue({ success: true });
    hooks.useChangeBillingControlPayer.mockReturnValue({ mutateAsync });
    renderPage();
    await user.selectOptions(screen.getByTestId('payer-select-101'), 'patient');
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          bookingId: 101,
          data: expect.objectContaining({ billing_intent: 'patient' }),
        }),
      );
    });
    expect(mockRefetch).toHaveBeenCalled();
  });

  it('U11 — Valider déclenche mutation', async () => {
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockResolvedValue({ success: true });
    hooks.useValidateBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /Valider/i }));
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({ bookingId: 101 }),
      );
    });
  });

  it('U12 — Anomalie ouvre modal et envoie motif', async () => {
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockResolvedValue({ success: true });
    hooks.useMarkBillingControlAnomaly.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /Signaler une anomalie/i }));
    expect(screen.getByText('Signaler une anomalie')).toBeInTheDocument();
    await user.type(screen.getByPlaceholderText(/Décrivez/i), 'Montant incorrect');
    await user.click(screen.getByRole('button', { name: /^Signaler$/i }));
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalled();
    });
  });

  it('U14 — booking verrouillé en lecture seule', () => {
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        items: [{
          booking_id: 999,
          scheduled_time: '2026-09-02T10:00:00',
          patient: { display_name: 'Locked' },
          segment_type: 'outbound',
          payer: { type: 'clinic' },
          control: { effective_status: 'validated' },
          billing: { editable: false, locked: true, invoiced: true },
        }],
        summary: { total: 1 },
        pagination: { page: 1, total_pages: 1, total: 1 },
      },
      isLoading: false,
      isError: false,
      refetch: mockRefetch,
      isFetching: false,
    });
    renderPage();
    expect(screen.getByText(/Facturé/)).toBeInTheDocument();
    expect(screen.queryByTestId('payer-select-999')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Réouvrir/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Valider/i })).not.toBeInTheDocument();
  });

  it.each([
    ['anomaly', 201],
    ['validated', 202],
  ])('U13 — Réouvrir visible sur %s et déclenche mutation', async (status, bookingId) => {
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        items: [{
          booking_id: bookingId,
          scheduled_time: '2026-09-02T10:00:00',
          patient: { display_name: status === 'anomaly' ? 'Ano' : 'Val' },
          segment_type: 'outbound',
          payer: { type: 'patient' },
          control: {
            effective_status: status,
            anomaly_reason: status === 'anomaly' ? 'OTHER: test' : undefined,
            validated_by_display_name: status === 'validated' ? 'Marc' : undefined,
          },
          billing: { editable: true, locked: false, invoiced: false },
        }],
        summary: { total: 1, anomaly: status === 'anomaly' ? 1 : 0, validated: status === 'validated' ? 1 : 0 },
        pagination: { page: 1, total_pages: 1, total: 1 },
      },
      isLoading: false,
      isError: false,
      refetch: mockRefetch,
      isFetching: false,
    });
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockResolvedValue({ success: true });
    hooks.useReopenBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /Réouvrir/i }));
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({ bookingId }),
      );
    });
  });

  it('U16 — 403 API affiché proprement', () => {
    hooks.useBillingControlBookings.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { response: { status: 403 } },
      refetch: mockRefetch,
      isFetching: false,
    });
    renderPage();
    expect(screen.getByTestId('billing-control-api-403')).toBeInTheDocument();
  });

  it('requester — accès refusé inline', () => {
    hooks.useInstitutionMe.mockReturnValue({
      data: { institution_role: 'institution_requester' },
    });
    renderPage();
    expect(screen.getByTestId('billing-control-forbidden')).toBeInTheDocument();
  });
});
