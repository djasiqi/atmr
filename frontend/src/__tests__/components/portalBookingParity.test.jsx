import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ClientDashboard from 'pages/client/Dashboard/ClientDashboard';
import apiClient from 'utils/apiClient';

const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => {
  const actual = jest.requireActual('react-router-dom');
  return { ...actual, useNavigate: () => mockNavigate };
});

jest.mock('utils/apiClient');
jest.mock('services/clientSaferpayPaymentService', () => ({
  startSaferpayHostedCheckout: jest.fn(() => Promise.resolve()),
}));
jest.mock('services/clientPortalSocket', () => ({
  ensureClientPortalSocket: jest.fn(() => Promise.resolve(null)),
  getClientPortalSocket: () => null,
  disconnectClientPortalSocket: jest.fn(),
}));
jest.mock('sonner', () => ({
  toast: { success: jest.fn(), error: jest.fn(), warning: jest.fn(), info: jest.fn() },
}));
jest.mock('@react-google-maps/api', () => ({
  GoogleMap: ({ children }) => <div data-testid="map-container">{children}</div>,
  Polyline: () => null,
}));
jest.mock('components/common/GoogleMapsAdvancedMarker', () => ({
  __esModule: true,
  default: () => null,
}));
jest.mock('components/common/GoogleMapsProvider', () => ({
  __esModule: true,
  default: ({ children }) => <>{children}</>,
  useGoogleMapsLoaded: () => ({ isLoaded: true, loadError: null }),
}));
jest.mock('components/layout/Header/HeaderDashboard', () => () => <div>Header</div>);
jest.mock('components/layout/Footer/Footer', () => () => <div>Footer</div>);
jest.mock('components/common/AddressAutocomplete', () => {
  return function MockAddressAutocomplete({ value, onChange, placeholder, inputId }) {
    return (
      <input
        id={inputId}
        data-testid={inputId || 'address-autocomplete'}
        value={value}
        onChange={(e) => onChange?.(e)}
        placeholder={placeholder}
      />
    );
  };
});

const previewBody = {
  data: {
    pricing: { amount: 40 },
    workflow: {},
    canonical_addresses: {
      pickup: {
        label: 'Rue du Test 1',
        canonical_hash: 'pickup-hash',
        precision_level: 'street',
      },
      dropoff: {
        label: 'HUG',
        canonical_hash: 'dropoff-hash',
        precision_level: 'street',
      },
    },
    booking_id: 1,
    booking: { id: 1, status: 'pending' },
  },
};

function renderDashboard() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <MemoryRouter initialEntries={['/dashboard/client/client-123']}>
      <QueryClientProvider client={queryClient}>
        <Routes>
          <Route path="/dashboard/client/:id" element={<ClientDashboard />} />
        </Routes>
      </QueryClientProvider>
    </MemoryRouter>
  );
}

async function fillAddresses() {
  const pickup = await screen.findByTestId('client-dashboard-pickup');
  await waitFor(() => {
    expect(screen.getByRole('button', { name: /Vérifier la demande/i })).toBeEnabled();
  });
  fireEvent.change(pickup, { target: { value: 'Rue du Test 1' } });
  fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
    target: { value: 'HUG' },
  });
  const service = screen.queryByLabelText(/^Service/i);
  const doctor = screen.queryByLabelText(/^Médecin/i);
  if (service) fireEvent.change(service, { target: { value: 'Radiologie' } });
  if (doctor) fireEvent.change(doctor, { target: { value: 'Dr Martin' } });
}

function isoToPickerDisplay(iso) {
  const [year, month, day] = String(iso).split('-');
  return `${day}.${month}.${year}`;
}

function plan(date = '2026-09-30', time = '09:00') {
  fireEvent.change(screen.getByLabelText('Date du transport'), {
    target: { value: isoToPickerDisplay(date) },
  });
  fireEvent.change(screen.getByLabelText('Heure du rendez-vous'), { target: { value: time } });
}

function planDeparture(date = '2026-09-30', time = '08:15') {
  fireEvent.change(screen.getByLabelText('Date du transport'), {
    target: { value: isoToPickerDisplay(date) },
  });
  fireEvent.change(screen.getByLabelText('Heure de départ'), { target: { value: time } });
}

function previewPayload() {
  const call = apiClient.post.mock.calls.find((entry) =>
    String(entry[0]).includes('/bookings/preview')
  );
  expect(call).toBeTruthy();
  return call[1];
}

describe('Parité formulaire PORTAL', () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    localStorage.setItem('authToken', 'fake-client-token');
    localStorage.setItem('public_id', 'client-123');
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({
          data: {
            id: 42,
            public_id: 'client-123',
            client_type: 'PORTAL',
            phone: '+41791234567',
            phone_verified: true,
            user: { first_name: 'Ada', last_name: 'Martin', birth_date: '1980-01-01' },
            domicile: { address: 'Rue de Lausanne 1, 1201 Genève', lat: 46.2, lon: 6.14 },
            billing_address: 'Rue de Lausanne 1, 1201 Genève',
          },
        });
      }
      if (String(url).includes('/bookings')) {
        return Promise.resolve({ data: [] });
      }
      return Promise.reject(new Error('Not found'));
    });
    apiClient.post.mockResolvedValue(previewBody);
  });

  it('exige l’heure de prise en charge ou le rendez-vous', async () => {
    renderDashboard();
    await fillAddresses();
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect(
      await screen.findByText('Indiquez l’heure de prise en charge ou l’heure du rendez-vous.')
    ).toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Récapitulatif de la demande' })
    ).not.toBeInTheDocument();
  });

  it('rendez-vous conserve l’heure sans la présenter comme une prise en charge', async () => {
    renderDashboard();
    await fillAddresses();
    plan();
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect((await screen.findAllByText(/Rendez-vous/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText('Prise en charge à déterminer')).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Prise en charge à 09:00/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
    await waitFor(() => expect(previewPayload().scheduled_time_type).toBe('arrival'));
    const body = previewPayload();
    expect(body.scheduled_time).toEqual(expect.any(String));
    expect(body.asap).toBe(false);
    expect(body.is_urgent).toBe(false);
  });

  it('heure de départ confirme la prise en charge', async () => {
    renderDashboard();
    await fillAddresses();
    planDeparture('2026-09-30', '08:15');
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect((await screen.findAllByText(/Prise en charge ·/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/08:15/)).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
    await waitFor(() => expect(previewPayload().scheduled_time_type).toBe('departure'));
  });

  it('aller-retour sans heure n’envoie pas return_time', async () => {
    renderDashboard();
    await fillAddresses();
    plan();
    fireEvent.click(screen.getByRole('button', { name: 'Autre jour' }));
    fireEvent.change(document.getElementById('client-booking-return-date'), {
      target: { value: '2026-10-15' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect((await screen.findAllByText(/Heure à définir/)).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
    await waitFor(() => expect(previewPayload().is_round_trip).toBe(true));
    const body = previewPayload();
    expect(body.return_date).toBe('2026-10-15');
    expect(body.return_time).toBeUndefined();
  });

  it('aller-retour avec heure envoie return_time', async () => {
    renderDashboard();
    await fillAddresses();
    plan();
    fireEvent.change(document.getElementById('client-booking-return-date'), {
      target: { value: '2026-09-30' },
    });
    fireEvent.change(document.getElementById('client-booking-return-time'), {
      target: { value: '16:30' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect((await screen.findAllByText(/Heure de départ/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/16:30/)).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
    await waitFor(() => expect(previewPayload().return_time).toEqual(expect.any(String)));
  });

  it('les fauteuils s’excluent et l’assistance se combine', async () => {
    renderDashboard();
    await fillAddresses();
    plan();
    fireEvent.click(screen.getByRole('button', { name: 'En fauteuil' }));
    fireEvent.click(screen.getByRole('button', { name: 'Fournir fauteuil' }));
    fireEvent.click(screen.getByRole('switch', { name: 'Assistance' }));
    fireEvent.change(screen.getByLabelText("Type d'assistance"), {
      target: { value: 'Aide à la marche' },
    });
    expect(screen.getByRole('button', { name: 'En fauteuil' })).toHaveAttribute(
      'aria-pressed',
      'false'
    );
    expect(screen.getByRole('button', { name: 'Fournir fauteuil' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByRole('switch', { name: 'Assistance' })).toHaveAttribute(
      'aria-checked',
      'true'
    );
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));
    expect(
      (await screen.findAllByText('Fauteuil à fournir · Assistance · Aide à la marche')).length
    ).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
    await waitFor(() => expect(previewPayload().wheelchair_need).toBe(true));
    const body = previewPayload();
    expect(body.wheelchair_client_has).toBe(false);
    expect(body.needs_assistance).toBe(true);
    expect(body.assistance_detail).toBe('Aide à la marche');
    expect(body.wheelchair_client_has && body.wheelchair_need).toBe(false);
  });
});
