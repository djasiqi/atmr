import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import InstitutionBillingControl from '../InstitutionBillingControl';
import { toast } from 'sonner';

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

const mockDecideDispute = jest.fn().mockResolvedValue({ success: true });
jest.mock('../../../../services/institutionBillingControlService', () => ({
  __esModule: true,
  default: {
    decideBillingControlDispute: (...args) => mockDecideDispute(...args),
  },
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

  it('U09/U10 — changement payeur affiche la valeur immédiatement sans refetch', async () => {
    const user = userEvent.setup();
    let resolveMut;
    const mutateAsync = jest.fn().mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    hooks.useChangeBillingControlPayer.mockReturnValue({ mutateAsync });
    renderPage();
    const select = screen.getByTestId('payer-select-101');
    expect(select).toHaveValue('clinic');
    await user.selectOptions(select, 'patient');
    expect(select).toHaveValue('patient');
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/0 Clinique/);
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/2 Patient/);
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        bookingId: 101,
        payerType: 'patient',
        data: expect.objectContaining({ billing_intent: 'patient' }),
      }),
    );
    expect(mockRefetch).not.toHaveBeenCalled();
    resolveMut({ success: true });
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });
    expect(mockRefetch).not.toHaveBeenCalled();
  });

  it('U09b — échec payeur : rollback + message', async () => {
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockRejectedValue(new Error('réseau'));
    hooks.useChangeBillingControlPayer.mockReturnValue({ mutateAsync });
    renderPage();
    const select = screen.getByTestId('payer-select-101');
    await user.selectOptions(select, 'patient');
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        "Le payeur n'a pas pu être modifié. Réessayez.",
      );
    });
    expect(select).toHaveValue('clinic');
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/1 Clinique/);
  });

  it('U09c — changement rapide : une réponse stale n’écrase pas le dernier choix', async () => {
    const user = userEvent.setup();
    const pending = [];
    const mutateAsync = jest.fn().mockImplementation(
      () => new Promise((resolve, reject) => { pending.push({ resolve, reject }); }),
    );
    hooks.useChangeBillingControlPayer.mockReturnValue({ mutateAsync });
    renderPage();
    const select = screen.getByTestId('payer-select-101');
    await user.selectOptions(select, 'patient');
    await user.selectOptions(select, 'clinic');
    expect(select).toHaveValue('clinic');
    expect(mutateAsync).toHaveBeenCalledTimes(2);
    pending[0].reject(new Error('stale'));
    await waitFor(() => {
      expect(select).toHaveValue('clinic');
    });
    expect(toast.error).not.toHaveBeenCalled();
    pending[1].resolve({ success: true });
  });

  it('U11 — Valider affiche Validé immédiatement sans refetch', async () => {
    const user = userEvent.setup();
    let resolveMut;
    const mutateAsync = jest.fn().mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    hooks.useValidateBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /✓ Valider/i }));

    const row = document.querySelector('[data-booking-id="101"]');
    expect(row).toHaveTextContent('✓ Validé');
    expect(row).not.toHaveTextContent('À vérifier');
    expect(screen.queryByRole('button', { name: /✓ Valider/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Signaler une anomalie/i })).not.toBeInTheDocument();
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/2 Validés/);
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/0 À vérifier/);
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ bookingId: 101 }),
    );
    expect(mockRefetch).not.toHaveBeenCalled();

    resolveMut({ success: true, control: { control_status: 'validated' } });
    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });
    expect(mockRefetch).not.toHaveBeenCalled();
  });

  it('U11b — échec validation : rollback ligne + compteurs + message', async () => {
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockRejectedValue(new Error('réseau'));
    hooks.useValidateBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /✓ Valider/i }));
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        "La validation n'a pas pu être enregistrée. Réessayez.",
      );
    });
    const row = document.querySelector('[data-booking-id="101"]');
    expect(row).toHaveTextContent('À vérifier');
    expect(screen.getByRole('button', { name: /✓ Valider/i })).toBeInTheDocument();
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/1 Validés/);
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/1 À vérifier/);
  });

  it('U11c — validations en série indépendantes', async () => {
    const user = userEvent.setup();
    const pendingCalls = [];
    const mutateAsync = jest.fn().mockImplementation(({ bookingId }) => (
      new Promise((resolve) => {
        pendingCalls.push({ bookingId, resolve });
      })
    ));
    hooks.useValidateBillingControlBooking.mockReturnValue({ mutateAsync });
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        ...listPayload,
        items: [
          listPayload.items[0],
          {
            ...listPayload.items[0],
            booking_id: 103,
            segment_type: 'return',
          },
        ],
        summary: {
          ...listPayload.summary,
          total: 2,
          validated: 0,
          pending_review: 2,
        },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: mockRefetch,
      isFetching: false,
    });
    renderPage();
    await user.click(screen.getAllByRole('button', { name: /✓ Valider/i })[0]);
    await user.click(screen.getAllByRole('button', { name: /✓ Valider/i })[0]);
    expect(mutateAsync).toHaveBeenCalledTimes(2);
    expect(document.querySelector('[data-booking-id="101"]')).toHaveTextContent('✓ Validé');
    expect(document.querySelector('[data-booking-id="103"]')).toHaveTextContent('✓ Validé');
    expect(mockRefetch).not.toHaveBeenCalled();
    pendingCalls.forEach(({ resolve }) => resolve({ success: true }));
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

  it('U13 — Réouvrir visible sur validated et déclenche mutation', async () => {
    const status = 'validated';
    const bookingId = 202;
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
    let resolveMut;
    const mutateAsync = jest.fn().mockImplementation(
      () => new Promise((resolve) => { resolveMut = resolve; }),
    );
    hooks.useReopenBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /Réouvrir/i }));
    const row = document.querySelector('[data-booking-id="202"]');
    expect(row).toHaveTextContent('À vérifier');
    expect(row).not.toHaveTextContent('✓ Validé');
    expect(screen.queryByRole('button', { name: /Réouvrir/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /✓ Valider/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Signaler une anomalie/i })).toBeInTheDocument();
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/0 Validés/);
    expect(screen.getByTestId('billing-control-summary').textContent.replace(/\s+/g, ' '))
      .toMatch(/1 À vérifier/);
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({ bookingId }),
    );
    expect(mockRefetch).not.toHaveBeenCalled();
    resolveMut({ success: true });
  });

  it('U13b — échec réouverture : rollback + message', async () => {
    const bookingId = 202;
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        items: [{
          booking_id: bookingId,
          scheduled_time: '2026-09-02T10:00:00',
          patient: { display_name: 'Val' },
          segment_type: 'outbound',
          payer: { type: 'patient' },
          control: {
            effective_status: 'validated',
            validated_by_display_name: 'Marc',
          },
          billing: { editable: true, locked: false, invoiced: false },
        }],
        summary: { total: 1, validated: 1, pending_review: 0 },
        pagination: { page: 1, total_pages: 1, total: 1 },
      },
      isLoading: false,
      isError: false,
      refetch: mockRefetch,
      isFetching: false,
    });
    const user = userEvent.setup();
    const mutateAsync = jest.fn().mockRejectedValue(new Error('réseau'));
    hooks.useReopenBillingControlBooking.mockReturnValue({ mutateAsync });
    renderPage();
    await user.click(screen.getByRole('button', { name: /Réouvrir/i }));
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        "La réouverture n'a pas pu être enregistrée. Réessayez.",
      );
    });
    expect(document.querySelector('[data-booking-id="202"]')).toHaveTextContent('✓ Validé');
    expect(screen.getByRole('button', { name: /Réouvrir/i })).toBeInTheDocument();
  });

  it('affiche Valider le justificatif quand une preuve est soumise', async () => {
    hooks.useBillingControlBookings.mockReturnValue({
      data: {
        items: [{
          booking_id: 45705,
          scheduled_time: '2026-08-16T10:00:00',
          patient: { display_name: 'Marie DUPONT' },
          segment_type: 'outbound',
          payer: { type: 'clinic' },
          control: {
            effective_status: 'anomaly',
            dispute_status: 'evidence_submitted',
            anomaly_reason: 'TRANSPORT_DISPUTED',
          },
          billing: { editable: true, locked: false, invoiced: false },
        }],
        summary: { total: 1, anomaly: 1 },
        pagination: { page: 1, total_pages: 1, total: 1 },
      },
      isLoading: false,
      isError: false,
      refetch: mockRefetch,
      isFetching: false,
    });
    const user = userEvent.setup();
    renderPage();
    expect(screen.queryByRole('button', { name: /Réouvrir/i })).not.toBeInTheDocument();
    await user.click(screen.getByTestId('dispute-accept-45705'));
    await waitFor(() => {
      expect(mockDecideDispute).toHaveBeenCalledWith(45705, { decision: 'accept_carrier' });
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
