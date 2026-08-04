import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import AdminBookingDetail, {
  formatRelativeAge,
  resolveBackPath,
} from '../AdminBookingDetail';
import { adminPaths } from '../../routing/adminRoutePaths';

jest.mock('../../../../services/adminService', () => ({
  fetchAdminBookingDetail: jest.fn(),
}));

const { fetchAdminBookingDetail } = require('../../../../services/adminService');

const detailPayload = {
  id: 3,
  transport: {
    status: 'accepted',
    status_label: 'Acceptée',
    scheduled_at: null,
    pickup: 'HUG Consultation',
    dropoff: 'Chemin des Courbes 9',
    amount_chf: 50,
    mission_type: 'patient_transport',
    is_round_trip: false,
    is_return: false,
    created_at: '2026-07-31T00:14:50+00:00',
    last_updated_at: '2026-07-31T00:14:50+00:00',
    last_updated_age_seconds: 420,
    edit_version: 1,
    cancelled_at: null,
  },
  support_diagnostic: {
    status: 'action_required',
    severity: 'blocking',
    needs_investigation: true,
    primary_reason_code: 'MISSING_SCHEDULED_TIME',
    headline: 'Horaire du transport manquant',
    summary:
      'Le transport est acceptée par Diaz, mais aucune date ni heure n\'est renseignée.',
    recommended_action: 'request_or_correct_schedule',
    reasons: [
      {
        code: 'MISSING_SCHEDULED_TIME',
        severity: 'blocking',
        label: 'Date et heure du transport manquantes',
        recommended_action: 'request_or_correct_schedule',
      },
      {
        code: 'MISSING_CREATOR',
        severity: 'info',
        label: 'Auteur de la création non identifié',
        recommended_action: null,
      },
    ],
  },
  actors: {
    client: { id: 23, label: 'Sofia GIUSEPPA' },
    requester: null,
    institution: null,
    current_company: { id: 3, label: 'Diaz' },
    executing_company: null,
    previous_company: null,
    driver: null,
    cancelled_by: null,
  },
  timeline: [
    {
      type: 'transport_created',
      at: '2026-07-31T00:14:50+00:00',
      label: 'Transport créé',
      detail: 'Auteur non identifié',
      actor: null,
      source: 'booking',
      details: null,
    },
  ],
  references: { booking_id: 3 },
};

const renderDetail = (initialEntry) =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route
          path="/dashboard/admin/:public_id/operations/bookings/:bookingId"
          element={<AdminBookingDetail />}
        />
      </Routes>
    </MemoryRouter>
  );

describe('AdminBookingDetail — console support', () => {
  beforeEach(() => {
    fetchAdminBookingDetail.mockReset();
  });

  it('affiche le diagnostic action requise et l’horaire manquant', async () => {
    fetchAdminBookingDetail.mockResolvedValue(detailPayload);
    renderDetail({
      pathname: '/dashboard/admin/pub-1/operations/bookings/3',
    });

    expect(await screen.findByRole('heading', { name: /Transport nº 3/i })).toBeInTheDocument();
    expect(
      (await screen.findAllByText(/Horaire du transport manquant/i)).length
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Action requise/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/Date et heure du transport manquantes/i)).toBeInTheDocument();
    expect(screen.getByText(/À définir/)).toBeInTheDocument();
    expect(screen.getAllByText(/Non affecté/).length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText(/Non applicable/i)).not.toBeInTheDocument();
    expect(
      screen.getByText(/Confirmer ou corriger l.horaire avec le demandeur/i)
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Entreprise Diaz/i })).toHaveAttribute(
      'href',
      expect.stringContaining('/partners/users?search=Diaz')
    );
    expect(screen.getByRole('link', { name: /Investigation technique/i })).toHaveAttribute(
      'href',
      expect.stringContaining('booking_id=3')
    );
  });

  it('préserve le retour vers les filtres via location.state.from', async () => {
    fetchAdminBookingDetail.mockResolvedValue(detailPayload);
    const from = '/dashboard/admin/pub-1/operations/bookings?needs_investigation=true';
    renderDetail({
      pathname: '/dashboard/admin/pub-1/operations/bookings/3',
      state: { from },
    });

    const back = await screen.findByRole('link', { name: /Retour aux transports/i });
    expect(back).toHaveAttribute('href', from);
  });

  it('affiche Attention pour un diagnostic warning', async () => {
    fetchAdminBookingDetail.mockResolvedValue({
      ...detailPayload,
      support_diagnostic: {
        status: 'attention',
        severity: 'warning',
        needs_investigation: false,
        primary_reason_code: 'MISSING_INSTITUTION',
        headline: 'Institution attendue non identifiée',
        summary: 'Avertissement.',
        recommended_action: null,
        reasons: [
          {
            code: 'MISSING_INSTITUTION',
            severity: 'warning',
            label: 'Institution attendue non identifiée',
            recommended_action: null,
          },
        ],
      },
      transport: { ...detailPayload.transport, scheduled_at: '2026-08-01T10:00:00+00:00' },
    });
    renderDetail({ pathname: '/dashboard/admin/pub-1/operations/bookings/3' });
    expect(
      (await screen.findAllByText(/Institution attendue non identifiée/i)).length
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/^Attention$/).length).toBeGreaterThanOrEqual(1);
  });
});

describe('helpers navigation / âge', () => {
  it('resolveBackPath valide le préfixe admin bookings', () => {
    const list = adminPaths.operationsBookings('pub-1');
    expect(
      resolveBackPath(
        { state: { from: `${list}?q=3` } },
        'pub-1',
        list
      )
    ).toBe(`${list}?q=3`);
    expect(
      resolveBackPath({ state: { from: '/evil' } }, 'pub-1', list)
    ).toBe(list);
  });

  it('formatRelativeAge', () => {
    expect(formatRelativeAge(30)).toMatch(/30 s/);
    expect(formatRelativeAge(420)).toMatch(/7 min/);
  });
});
