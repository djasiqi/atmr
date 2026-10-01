import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import WorkTimePanel from './WorkTimePanel';

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  useLocation: () => ({ pathname: '/dashboard/company/1/drivers' }),
  useNavigate: () => mockNavigate,
}));

jest.mock('../../../../services/companyService', () => ({
  fetchWorkTimeSummary: jest.fn(),
  fetchDriverWorkTime: jest.fn(),
  fetchBookingWorkTimeExplain: jest.fn(),
  createWorkTimeAdjustment: jest.fn(),
  createManualWorkEntry: jest.fn(),
  cancelManualWorkEntry: jest.fn(),
  finalizeWorkTimePeriod: jest.fn(),
  reopenWorkTimePeriod: jest.fn(),
}));

const { fetchWorkTimeSummary } = require('../../../../services/companyService');

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <WorkTimePanel />
    </QueryClientProvider>
  );
}

test('le tableau suit les transports et le mode d’affichage', async () => {
  fetchWorkTimeSummary.mockResolvedValue({
    period: { from: '2026-09-01', to: '2026-09-30' },
    period_finalized: false,
    contractual_rules: {
      version_count: 1,
      versions: [
        {
          policy_id: 1,
          mode: 'flat_per_trip',
          transport_flat_minutes: 30,
          effective_from: '2026-01-01',
          effective_until: null,
        },
      ],
    },
    kpis: {
      transport_count: 2,
      real_transport_minutes: 40,
      flat_transport_minutes: 60,
      real_added_minutes: 20,
      flat_added_minutes: 0,
      review_count_real: 1,
      review_count_flat: 1,
    },
    review_items: [
      {
        driver_id: 7,
        modes: ['real', 'flat'],
        reasons: ['arrival_not_recorded'],
        date: '2026-09-28',
        pickup_label: 'HUG',
        dropoff_label: 'Pictet',
      },
    ],
    drivers: [
      {
        driver_id: 7,
        display_name: 'Sod ERDENE',
        transport_count: 2,
        real_transport_minutes: 40,
        flat_transport_minutes: 60,
        real_added_minutes: 20,
        flat_added_minutes: 0,
        review_count_real: 1,
        review_count_flat: 1,
      },
    ],
  });

  renderPanel();
  expect(await screen.findByText('Sod ERDENE')).toBeTruthy();
  expect(screen.getByText('Affichage')).toBeTruthy();
  expect(screen.getByText('Règle de clôture')).toBeTruthy();
  expect(screen.getByText('Forfait par transport · 30 min')).toBeTruthy();
  expect(screen.getAllByText('Transports').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Temps transport').length).toBeGreaterThan(0);
  expect(screen.getAllByText('Temps ajouté').length).toBeGreaterThan(0);
  expect(screen.queryByText('Résumé des heures — Sod ERDENE')).toBeNull();
  expect(screen.getByText('1 élément à vérifier')).toBeTruthy();
  expect(screen.queryByText('Temps rémunéré')).toBeNull();
  expect(screen.queryByText(/segment/i)).toBeNull();

  expect(screen.getAllByText('0h40').length).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole('button', { name: '1 élément à vérifier' }));
  expect(screen.getByRole('heading', { name: 'À vérifier — 1 élément' })).toBeTruthy();
  expect(screen.getByText('Arrivée non enregistrée')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Examiner' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '← Tous les chauffeurs' }));
  fireEvent.click(screen.getByRole('button', { name: 'Forfait entreprise' }));
  expect(screen.queryByText('0h40')).toBeNull();
  expect(screen.getAllByText('1h00').length).toBeGreaterThan(0);
});

test('Ouvrir la course appelle le panneau local sans quitter la page', async () => {
  const { fetchDriverWorkTime } = require('../../../../services/companyService');
  const onOpenBooking = jest.fn();
  mockNavigate.mockClear();

  fetchWorkTimeSummary.mockResolvedValue({
    period: { from: '2026-10-01', to: '2026-10-31' },
    period_finalized: false,
    kpis: { review_count_real: 0, review_count_flat: 0 },
    drivers: [{ driver_id: 7, display_name: 'Inès DUBOIS', transport_count: 1 }],
  });
  fetchDriverWorkTime.mockResolvedValue({
    display_name: 'Inès DUBOIS',
    total_days: 1,
    days: [
      {
        date: '2026-10-01',
        entries: [
          {
            kind: 'transport',
            booking_id: 46793,
            work_time_status: 'verified',
            real_minutes: 14,
            pickup_label: 'HUG',
            dropoff_label: 'Avenue Ernest-Pictet 9',
          },
        ],
      },
    ],
  });

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkTimePanel onOpenBooking={onOpenBooking} />
    </QueryClientProvider>
  );

  fireEvent.click(await screen.findByRole('button', { name: 'Résumé des heures de Inès DUBOIS' }));
  fireEvent.click(await screen.findByRole('button', { name: '1 oct.' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Autres actions' }));
  fireEvent.click(screen.getByRole('menuitem', { name: 'Ouvrir la course' }));

  expect(onOpenBooking).toHaveBeenCalledWith(
    46793,
    expect.objectContaining({
      booking_id: 46793,
      pickup_label: 'HUG',
      dropoff_label: 'Avenue Ernest-Pictet 9',
    })
  );
  expect(mockNavigate).not.toHaveBeenCalled();
});

test('ajouter un temps liste les chauffeurs actifs et affiche la durée', async () => {
  fetchWorkTimeSummary.mockResolvedValue({
    period: { from: '2026-10-01', to: '2026-10-31' },
    period_finalized: false,
    kpis: { review_count_real: 0, review_count_flat: 0 },
    drivers: [{ driver_id: 7, display_name: 'Inès DUBOIS' }],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkTimePanel
        companyDrivers={[
          { id: 7, first_name: 'Inès', last_name: 'DUBOIS', is_active: true },
          { id: 8, first_name: 'Léa', last_name: 'MARTIN', is_active: true },
          { id: 9, first_name: 'Inactif', last_name: 'X', is_active: false },
        ]}
      />
    </QueryClientProvider>
  );

  fireEvent.click(await screen.findByRole('button', { name: 'Ajouter un temps' }));
  expect(screen.getByRole('dialog', { name: 'Ajouter un temps' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Chauffeur' }));
  expect(screen.getByRole('option', { name: 'Léa MARTIN' })).toBeTruthy();
  expect(screen.getByRole('option', { name: 'Inès DUBOIS' })).toBeTruthy();
  expect(screen.queryByRole('option', { name: 'Inactif X' })).toBeNull();
  expect(screen.getByText('1h00')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Enregistrer' })).toBeDisabled();

  fireEvent.change(screen.getByLabelText('Heure de fin'), { target: { value: '0700' } });
  expect(screen.getByText('Fin avant le début')).toBeTruthy();
});
