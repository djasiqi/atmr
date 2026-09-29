// frontend/tests/components/ClientDashboard.test.jsx
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ClientDashboard from 'pages/client/Dashboard/ClientDashboard';
import apiClient from 'utils/apiClient';
import { startSaferpayHostedCheckout } from 'services/clientSaferpayPaymentService';

const mockNavigate = jest.fn();

function isoToPickerDisplay(iso) {
  const [year, month, day] = String(iso).split('-');
  return `${day}.${month}.${year}`;
}

function setTransportDate(iso) {
  fireEvent.change(screen.getByLabelText('Date du transport'), {
    target: { value: isoToPickerDisplay(iso) },
  });
}

function readTransportDateIso() {
  const raw = document.getElementById('client-booking-date')?.value || '';
  const dotted = raw.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (dotted) return `${dotted[3]}-${dotted[2]}-${dotted[1]}`;
  return raw;
}

jest.mock('react-router-dom', () => {
  const actual = jest.requireActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mocks
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
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    warning: jest.fn(),
    info: jest.fn(),
  },
}));

// Mock @react-google-maps/api
jest.mock('@react-google-maps/api', () => ({
  GoogleMap: ({ children }) => <div data-testid="map-container">{children}</div>,
  Polyline: () => null,
}));

jest.mock('components/common/GoogleMapsAdvancedMarker', () => ({
  __esModule: true,
  default: () => null,
}));

// Mock GoogleMapsProvider
jest.mock('components/common/GoogleMapsProvider', () => ({
  __esModule: true,
  default: ({ children }) => <>{children}</>,
  useGoogleMapsLoaded: () => ({ isLoaded: true, loadError: null }),
}));

// Mock layout components
jest.mock('components/layout/Header/HeaderDashboard', () => {
  return function MockHeaderDashboard() {
    return <div data-testid="header-dashboard">Header</div>;
  };
});

jest.mock('components/layout/Footer/Footer', () => {
  return function MockFooter() {
    return <div data-testid="footer">Footer</div>;
  };
});

jest.mock('components/common/AddressAutocomplete', () => {
  return function MockAddressAutocomplete({ value, onChange, onSelect, placeholder, inputId }) {
    return (
      <div>
        <input
          id={inputId}
          data-testid={inputId || 'address-autocomplete'}
          value={value}
          onChange={(e) => onChange?.(e)}
          placeholder={placeholder}
        />
        <button
          type="button"
          data-testid={`${inputId}-select`}
          onClick={() =>
            onSelect?.({
              label: `${placeholder} validée`,
              lat: 46.2044,
              lon: 6.1432,
            })
          }
        >
          select
        </button>
      </div>
    );
  };
});

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }) => (
    <MemoryRouter initialEntries={['/dashboard/client/client-123']}>
      <QueryClientProvider client={queryClient}>
        <Routes>
          <Route path="/dashboard/client/:id" element={children} />
        </Routes>
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('ClientDashboard', () => {
  const now = Date.now();
  const toIso = (msFromNow) => new Date(now + msFromNow).toISOString();
  const mockProfile = {
    id: 42,
    public_id: 'client-123',
    client_type: 'PORTAL',
    user: {
      first_name: 'Jean',
      last_name: 'Dupont',
      email: 'jean.dupont@example.com',
    },
    billing_address: 'Rue de Lausanne 1, 1201 Genève',
  };

  const previewAndCreateResponse = (booking) => ({
    data: {
      pricing: { amount: 90 },
      workflow: { payment_required: true },
      canonical_addresses: {
        pickup: {
          label: 'Genève Gare',
          canonical_hash: 'pickup-hash',
          precision_level: 'address',
        },
        dropoff: {
          label: 'Lausanne Gare',
          canonical_hash: 'dropoff-hash',
          precision_level: 'address',
        },
      },
      data: {
        booking_id: booking.id,
        trace_id: 'trace',
        booking,
      },
    },
  });

  const mockBookings = [];

  const dismissEmptyMedicalDestination = () => {
    const medicalSwitch = screen.queryByRole('switch', { name: 'Destination médicale' });
    if (!medicalSwitch || medicalSwitch.getAttribute('aria-checked') !== 'true') return;
    const facility = screen.queryByLabelText(/Établissement/i);
    const service = screen.queryByLabelText(/^Service/i);
    const doctor = screen.queryByLabelText(/^Médecin/i);
    if (facility?.value || service?.value || doctor?.value) return;
    fireEvent.click(medicalSwitch);
  };

  const openPortalReview = async () => {
    const returnDay = document.getElementById('client-booking-return-date');
    if (returnDay) {
      const outboundValue = readTransportDateIso();
      if (!returnDay.value || (outboundValue && returnDay.value < outboundValue)) {
        const base = outboundValue
          ? new Date(`${outboundValue}T12:00:00`)
          : new Date(Date.now() + 24 * 60 * 60 * 1000);
        base.setDate(base.getDate() + 1);
        const y = base.getFullYear();
        const m = String(base.getMonth() + 1).padStart(2, '0');
        const d = String(base.getDate()).padStart(2, '0');
        fireEvent.change(returnDay, { target: { value: `${y}-${m}-${d}` } });
      }
    }
    dismissEmptyMedicalDestination();
    const verify = await screen.findByRole('button', { name: /Vérifier la demande/i });
    await waitFor(() => expect(verify).toBeEnabled());
    fireEvent.click(verify);
    expect(
      await screen.findByRole('heading', { name: 'Récapitulatif de la demande' })
    ).toBeInTheDocument();
  };

  const confirmPortalOrder = async () => {
    await openPortalReview();
    fireEvent.click(screen.getByRole('button', { name: 'Confirmer la demande de transport' }));
  };

  afterEach(() => {
    jest.useRealTimers();
  });

  beforeEach(() => {
    jest.useRealTimers();
    mockNavigate.mockClear();
    jest.clearAllMocks();
    window.__LIRIE_CLIENT_KPI__ = [];
    localStorage.clear();
    sessionStorage.clear();
    localStorage.setItem('authToken', 'fake-client-token');
    localStorage.setItem('public_id', 'client-123');
    // Mock profil client
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({ data: mockBookings });
      }
      return Promise.reject(new Error('Not found'));
    });
    apiClient.post.mockResolvedValue(
      previewAndCreateResponse({
        id: 999,
        amount: 50,
        billed_to_type: 'patient',
        status: 'pending',
        pickup_location: 'A',
        dropoff_location: 'B',
      })
    );
  });

  afterEach(() => {
    jest.useRealTimers();
    localStorage.clear();
  });

  it('devrait afficher le dashboard client', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByTestId('header-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('footer')).toBeInTheDocument();
  });

  it('devrait charger le profil du client', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledWith(
        '/clients/client-123',
        expect.objectContaining({
          headers: { Authorization: 'Bearer fake-client-token' },
        })
      );
    });
  });

  it("préremplit le lieu de prise en charge avec l'adresse domicile du profil", async () => {
    localStorage.setItem(
      'client:lastBooking:client-123',
      JSON.stringify({
        pickup: 'Ancien départ',
        destination: 'HUG — ne doit pas être restauré',
        status: 'En attente',
      })
    );
    render(<ClientDashboard />, { wrapper: createWrapper() });
    const pickupInput = await screen.findByTestId('client-dashboard-pickup');
    const dropoffInput = await screen.findByTestId('client-dashboard-dropoff');
    await waitFor(() => {
      expect(pickupInput).toHaveValue('Rue de Lausanne 1, 1201 Genève');
      expect(dropoffInput).toHaveValue('');
    });
    const medicalSwitch = screen.getByRole('switch', { name: 'Destination médicale' });
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'false');
    expect(screen.queryByLabelText(/Établissement/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Service$/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Médecin$/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Préciser l’établissement/i })).not.toBeInTheDocument();
    fireEvent.click(medicalSwitch);
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByLabelText(/Établissement/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Service/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Médecin/i)).toBeInTheDocument();
  });

  it('active la destination médicale quand le lieu est un cabinet ou un dentiste', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    const dropoffInput = await screen.findByTestId('client-dashboard-dropoff');
    fireEvent.change(dropoffInput, { target: { value: 'Gare de Lausanne' } });
    const medicalSwitch = screen.getByRole('switch', { name: 'Destination médicale' });
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'false');

    fireEvent.change(dropoffInput, { target: { value: 'Dentiste des Eaux-Vives' } });
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByLabelText(/Établissement/i)).toHaveValue('Dentiste des Eaux-Vives');
    expect(screen.getByLabelText(/^Service/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^Médecin/i)).toBeInTheDocument();
  });

  it('range un docteur dans Médecin et propose Cabinet médical', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    fireEvent.change(await screen.findByTestId('client-dashboard-dropoff'), {
      target: { value: 'Dr méd. Bigler Jean-Michel' },
    });
    expect(screen.getByLabelText(/Établissement/i)).toHaveValue('Cabinet médical');
    expect(screen.getByLabelText(/^Médecin/i)).toHaveValue('Dr méd. Bigler Jean-Michel');
  });

  it('éteint la destination médicale automatique quand la destination est vidée', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    const dropoffInput = await screen.findByTestId('client-dashboard-dropoff');
    fireEvent.change(dropoffInput, { target: { value: 'Clinique de Carouge' } });
    const medicalSwitch = screen.getByRole('switch', { name: 'Destination médicale' });
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'true');

    fireEvent.change(dropoffInput, { target: { value: '' } });
    expect(medicalSwitch).toHaveAttribute('aria-checked', 'false');
    expect(screen.queryByLabelText(/Établissement/i)).not.toBeInTheDocument();
  });

  it('émet les événements KPI clés de réservation', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    await screen.findByTestId('client-dashboard-pickup');
    expect(window.__LIRIE_CLIENT_KPI__.some((e) => e.name === 'reserve_opened')).toBe(true);

    fireEvent.change(screen.getByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Vérifier la demande/i }));

    await waitFor(() => {
      expect(window.__LIRIE_CLIENT_KPI__.some((e) => e.name === 'reserve_cta_clicked')).toBe(true);
    });
  });

  it('affiche la carte en arrière-plan dès l’API Google chargée (texte libre sans validation)', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    expect(await screen.findByTestId('map-container')).toBeInTheDocument();
    fireEvent.change(screen.getByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });
    expect(screen.getByTestId('map-container')).toBeInTheDocument();
  });

  it('affiche la carte après sélection autocomplete validée', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    fireEvent.click(await screen.findByTestId('client-dashboard-pickup-select'));
    fireEvent.click(screen.getByTestId('client-dashboard-dropoff-select'));
    expect(await screen.findByTestId('map-container')).toBeInTheDocument();
  });

  it('affiche le bloc prochaine course avec actions selon statut', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 1,
              pickup_location: 'Genève',
              dropoff_location: 'Lausanne',
              scheduled_time: toIso(2 * 60 * 60 * 1000),
              status: 'CONFIRMED',
              amount: 50,
              is_round_trip: true,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Prochaine course/i)).toBeInTheDocument();
    expect(screen.getByText(/Course confirmée/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Voir/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Modifier/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Annuler/i })).toBeInTheDocument();
    expect(screen.getByText('Aller-retour', { selector: '.bookingTripKindChip' })).toBeInTheDocument();
  });

  it('affiche la pastille Retour pour une course retour (is_return)', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 2,
              pickup_location: 'Lausanne',
              dropoff_location: 'Genève',
              scheduled_time: toIso(3 * 60 * 60 * 1000),
              status: 'CONFIRMED',
              amount: 50,
              is_return: true,
              is_round_trip: false,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Prochaine course/i)).toBeInTheDocument();
    expect(screen.getByText('Retour', { selector: '.bookingTripKindChip' })).toBeInTheDocument();
    expect(
      screen.queryByText('Aller-retour', { selector: '.bookingTripKindChip' })
    ).not.toBeInTheDocument();
  });

  it('ne place pas un retour déjà terminé dans Prochaine course', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 46799,
              pickup_location: 'Clinique de Joli-Mont, Avenue Trembley 45, 1209, Genève',
              dropoff_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
              scheduled_time: toIso(20 * 60 * 60 * 1000),
              status: 'return_completed',
              amount: 40,
              is_return: true,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Reprendre un trajet récent/i)).toBeInTheDocument();
    expect(screen.queryByText(/Prochaine course/i)).not.toBeInTheDocument();
    expect(screen.queryByText('Terminée')).not.toBeInTheDocument();
  });

  it('affiche les courses recentes quand il n y a pas de course active/future', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 21,
              pickup_location: 'Vevey',
              dropoff_location: 'Montreux',
              scheduled_time: toIso(-24 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 35,
              has_return: true,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Reprendre un trajet récent/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Réutiliser ce trajet/i })).toBeInTheDocument();
    expect(screen.getByText('Aller-retour', { selector: '.bookingTripKindChip' })).toBeInTheDocument();
  });

  it('regroupe un transport à plusieurs étapes en un seul trajet récent', async () => {
    const groupId = '18b0975a-221b-455c-a526-f5574b4122fc';
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 46797,
              route_group_id: groupId,
              route_sequence_number: 1,
              pickup_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
              dropoff_location: 'Hôpitaux Universitaires de Genève (HUG)',
              hospital_service: 'Radiologie',
              scheduled_time: toIso(-3 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 40,
              is_round_trip: true,
              is_return: false,
              company_id: 1,
            },
            {
              id: 46798,
              route_group_id: groupId,
              route_sequence_number: 2,
              pickup_location: 'Hôpitaux Universitaires de Genève (HUG)',
              dropoff_location: 'Clinique de Joli-Mont',
              doctor_name: 'Docteur Rashiti',
              scheduled_time: toIso(-2 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 40,
              is_round_trip: false,
              is_return: false,
              company_id: 1,
            },
            {
              id: 46799,
              route_group_id: groupId,
              route_sequence_number: 3,
              parent_booking_id: 46798,
              pickup_location: 'Clinique de Joli-Mont',
              dropoff_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
              scheduled_time: null,
              status: 'ACCEPTED',
              amount: 40,
              is_round_trip: false,
              is_return: true,
              company_id: 1,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Reprendre un trajet récent/i)).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Réutiliser ce trajet/i })).toHaveLength(1);
    expect(screen.getByText('3 trajets', { selector: '.bookingTripKindChip' })).toBeInTheDocument();
    expect(screen.queryByText('Aller-retour', { selector: '.bookingTripKindChip' })).not.toBeInTheDocument();
    expect(screen.getByText('Étape 1')).toBeInTheDocument();
    expect(screen.getByText(/Hôpitaux Universitaires de Genève/)).toBeInTheDocument();
    expect(screen.getByText('Étape 2')).toBeInTheDocument();
    expect(screen.getByText('Clinique de Joli-Mont')).toBeInTheDocument();
    expect(screen.getByText('Retour', { selector: '.recentTripLegLabel' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Réutiliser ce trajet/i }));

    expect(screen.getByTestId('client-dashboard-pickup')).toHaveValue(
      'Avenue Ernest-Pictet 9, 1203, Genève'
    );
    expect(screen.getByTestId('client-dashboard-dropoff')).toHaveValue(
      'Hôpitaux Universitaires de Genève (HUG)'
    );
    expect(screen.getByLabelText('Étape 2')).toHaveValue('Clinique de Joli-Mont');
  });

  it('propose le dernier trajet et le trajet le plus utilisé', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: [
            {
              id: 10,
              pickup_location: 'Avenue Ernest-Pictet 9',
              dropoff_location: 'HUG',
              scheduled_time: toIso(-20 * 24 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 40,
            },
            {
              id: 11,
              pickup_location: 'Avenue Ernest-Pictet 9',
              dropoff_location: 'HUG',
              scheduled_time: toIso(-10 * 24 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 40,
            },
            {
              id: 12,
              pickup_location: 'Genève',
              dropoff_location: 'Lausanne',
              scheduled_time: toIso(-2 * 60 * 60 * 1000),
              status: 'COMPLETED',
              amount: 50,
            },
          ],
        });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByText('Dernier')).toBeInTheDocument();
    expect(screen.getByText('Le plus utilisé')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Réutiliser ce trajet/i })).toHaveLength(2);
    expect(screen.getByText('Lausanne')).toBeInTheDocument();
    expect(screen.getByText('HUG')).toBeInTheDocument();
  });

  it('reprend un aller simple sans remettre le retour', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    render(
      <MemoryRouter
        initialEntries={[
          {
            pathname: '/dashboard/client/client-123',
            state: {
              prefillFromBooking: {
                pickup_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
                dropoff_location:
                  'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
                dropoff_detail: 'Radiologie',
                extra_stops: [],
                round_trip: false,
              },
            },
          },
        ]}
      >
        <QueryClientProvider client={queryClient}>
          <Routes>
            <Route path="/dashboard/client/:id" element={<ClientDashboard />} />
          </Routes>
        </QueryClientProvider>
      </MemoryRouter>
    );

    expect(await screen.findByLabelText('Aller-retour')).toHaveAttribute('aria-checked', 'false');
    expect(screen.queryByText('Identique au point de départ')).not.toBeInTheDocument();
    expect(screen.getByTestId('client-dashboard-pickup')).toHaveValue(
      'Avenue Ernest-Pictet 9, 1203, Genève'
    );
    await waitFor(() => {
      expect(document.getElementById('client-booking-service')).toHaveValue('Radiologie');
    });
  });

  it('affiche les horaires du trajet et la date', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByLabelText('Heure de départ')).toBeEnabled();
    expect(screen.getByLabelText('Heure du rendez-vous')).toBeEnabled();
    expect(screen.getByLabelText(/Date du transport/i)).toBeEnabled();
  });

  it('ouvre le calendrier au clic sur la date de l’en-tête', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    const dateField = await screen.findByLabelText(/Date du transport/i);

    fireEvent.click(dateField.closest('.portalHeaderDate'));

    expect(await screen.findByRole('dialog', { name: 'Choisir une date' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Mois précédent' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Mois suivant' }));
    expect(screen.getByRole('dialog', { name: 'Choisir une date' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Mois précédent' })).toBeEnabled();

    fireEvent.click(screen.getByRole('button', { name: 'Mois précédent' }));
    expect(screen.getByRole('button', { name: 'Mois précédent' })).toBeDisabled();
  });

  it('la réservation reste possible si estimation itinéraire échoue', async () => {
    apiClient.post.mockImplementation((url) => {
      if (url === '/ai/optimized-route') {
        return Promise.reject(new Error('route-failed'));
      }
      if (url.includes('/bookings')) {
        return Promise.resolve(
          previewAndCreateResponse({
            id: 777,
            pickup_location: 'Genève',
            dropoff_location: 'Lausanne',
            scheduled_time: toIso(2 * 60 * 60 * 1000),
            status: 'pending',
            amount: 50,
            billed_to_type: 'patient',
          })
        );
      }
      return Promise.reject(new Error('unknown-post'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    fireEvent.change(await screen.findByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });

    expect(
      await screen.findByText(/Impossible d’estimer ce trajet pour le moment/i, {}, { timeout: 4000 })
    ).toBeInTheDocument();

    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    setTransportDate(
      `${tomorrow.getFullYear()}-${String(tomorrow.getMonth() + 1).padStart(2, '0')}-${String(tomorrow.getDate()).padStart(2, '0')}`
    );
    fireEvent.change(screen.getByLabelText('Heure de départ'), { target: { value: '10:30' } });

    await confirmPortalOrder();

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/clients/client-123/bookings',
        expect.objectContaining({
          pickup_location: 'Genève Gare',
          dropoff_location: 'Lausanne Gare',
        }),
        expect.any(Object)
      );
    });
  });

  it('ne lance pas Saferpay pour un compte PORTAL même si billed_to_type est patient', async () => {
    startSaferpayHostedCheckout.mockClear();
    render(<ClientDashboard />, { wrapper: createWrapper() });

    fireEvent.change(await screen.findByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });

    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const y = tomorrow.getFullYear();
    const m = String(tomorrow.getMonth() + 1).padStart(2, '0');
    const d = String(tomorrow.getDate()).padStart(2, '0');
    setTransportDate(`${y}-${m}-${d}`);
    fireEvent.change(screen.getByLabelText('Heure du rendez-vous'), {
      target: { value: '10:30' },
    });
    await confirmPortalOrder();

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/clients/client-123/bookings',
        expect.any(Object),
        expect.any(Object)
      );
    });
    expect(startSaferpayHostedCheckout).not.toHaveBeenCalled();
    expect(screen.queryByText('Paiement sécurisé')).not.toBeInTheDocument();
  });

  it('lance Saferpay pour un client TRANSPORT lorsque billed_to_type est patient', async () => {
    startSaferpayHostedCheckout.mockClear();
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({
          data: { ...mockProfile, client_type: 'TRANSPORT' },
        });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({ data: mockBookings });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });

    await screen.findByTestId('client-dashboard-pickup');

    fireEvent.change(await screen.findByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });

    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const y = tomorrow.getFullYear();
    const m = String(tomorrow.getMonth() + 1).padStart(2, '0');
    const d = String(tomorrow.getDate()).padStart(2, '0');
    setTransportDate(`${y}-${m}-${d}`);
    fireEvent.change(screen.getByLabelText('Heure du rendez-vous'), {
      target: { value: '10:30' },
    });
    const returnLater = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000);
    fireEvent.change(document.getElementById('client-booking-return-date'), {
      target: {
        value: `${returnLater.getFullYear()}-${String(returnLater.getMonth() + 1).padStart(2, '0')}-${String(returnLater.getDate()).padStart(2, '0')}`,
      },
    });
    dismissEmptyMedicalDestination();
    const submitTransport = await screen.findByRole('button', {
      name: /Valider la demande de transport/i,
    });
    await waitFor(() => expect(submitTransport).toBeEnabled());
    fireEvent.click(submitTransport);

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/clients/client-123/bookings',
        expect.any(Object),
        expect.any(Object)
      );
      expect(startSaferpayHostedCheckout).toHaveBeenCalledWith(999);
    });
  });

  it('ne lance pas Saferpay pour une réservation tiers payeur (assurance)', async () => {
    startSaferpayHostedCheckout.mockClear();
    apiClient.post.mockResolvedValue(
      previewAndCreateResponse({
        id: 1002,
        amount: 50,
        billed_to_type: 'insurance',
        status: 'pending',
      })
    );

    render(<ClientDashboard />, { wrapper: createWrapper() });

    fireEvent.change(await screen.findByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });

    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const y = tomorrow.getFullYear();
    const m = String(tomorrow.getMonth() + 1).padStart(2, '0');
    const d = String(tomorrow.getDate()).padStart(2, '0');
    setTransportDate(`${y}-${m}-${d}`);
    fireEvent.change(screen.getByLabelText('Heure du rendez-vous'), {
      target: { value: '10:30' },
    });
    await confirmPortalOrder();

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/clients/client-123/bookings',
        expect.any(Object),
        expect.any(Object)
      );
    });

    expect(startSaferpayHostedCheckout).not.toHaveBeenCalled();
  });

  const fillOutbound = async () => {
    fireEvent.change(await screen.findByTestId('client-dashboard-pickup'), {
      target: { value: 'Genève Gare' },
    });
    fireEvent.change(screen.getByTestId('client-dashboard-dropoff'), {
      target: { value: 'Lausanne Gare' },
    });
    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const y = tomorrow.getFullYear();
    const m = String(tomorrow.getMonth() + 1).padStart(2, '0');
    const d = String(tomorrow.getDate()).padStart(2, '0');
    setTransportDate(`${y}-${m}-${d}`);
    fireEvent.change(document.getElementById('client-booking-time'), {
      target: { value: '10:30' },
    });
  };

  it('affiche une estimation indicative et le titulaire comme débiteur', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    await fillOutbound();
    await openPortalReview();

    expect(screen.getByText(/Contact : Jean Dupont/)).toBeInTheDocument();
    // Hors DV : estimation indicative. Avec DV : uniquement le plafond (pas les deux).
    const hasEstimate = screen.queryByText(/Estimation indicative : CHF/i);
    const hasCeiling = screen.queryByText(/Prix maximum accepté \(pas le prix final\)/i);
    expect(hasEstimate || hasCeiling).toBeTruthy();
    if (hasCeiling) {
      expect(screen.queryByText(/Estimation indicative : CHF/i)).not.toBeInTheDocument();
      expect(
        screen.getByText(/Ce plafond n’est pas le prix à payer/i)
      ).toBeInTheDocument();
    } else {
      expect(screen.getByText(/Indicative — le montant final/i)).toBeInTheDocument();
    }
    expect(
      screen.getByRole('button', { name: 'Confirmer la demande de transport' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /CHF/i })).not.toBeInTheDocument();
    expect(
      screen.getByText(/Aucune acceptation des conditions n’est enregistrée/i)
    ).toBeInTheDocument();
    expect(apiClient.post).not.toHaveBeenCalled();
  });

  it('ne crée qu’une demande et réutilise la clé d’idempotence', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    await fillOutbound();
    await openPortalReview();
    const confirm = screen.getByRole('button', { name: 'Confirmer la demande de transport' });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    await waitFor(() => {
      const creates = apiClient.post.mock.calls.filter(
        (call) => String(call[0]).includes('/bookings') && !String(call[0]).includes('preview')
      );
      expect(creates).toHaveLength(1);
      expect(creates[0][2].headers['Idempotency-Key']).toEqual(expect.any(String));
    });
  });

  it('affiche le corps canonique servi par le catalogue', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (String(url).includes('/clients/me/portal-terms')) {
        return Promise.resolve({
          data: {
            data: [
              {
                document_type: 'terms_of_service',
                terms_version: '1.0',
                canonical_body: 'CORPS CANONIQUE CGU',
              },
              {
                document_type: 'transport_terms',
                terms_version: '1.0',
                canonical_body: 'CORPS CANONIQUE CGV',
              },
            ],
          },
        });
      }
      if (String(url).includes('/clients/me/terms-acceptances')) {
        return Promise.resolve({
          data: {
            data: [
              { document_type: 'terms_of_service' },
              { document_type: 'transport_terms' },
            ],
          },
        });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({ data: mockBookings });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });
    await fillOutbound();
    await openPortalReview();
    expect(
      await screen.findByText(/conditions acceptées pour votre compte/i)
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole('button', { name: /Conditions générales d’utilisation 1.0/i })
    );
    expect(await screen.findByText('CORPS CANONIQUE CGU')).toBeInTheDocument();
  });

  it('demande une acceptation explicite avant une nouvelle réservation', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (String(url).includes('/clients/me/portal-terms-status')) {
        return Promise.resolve({
          data: {
            data: {
              status: 'reacceptance_required',
              documents: [
                {
                  document_type: 'transport_terms',
                  current_version: '2.0',
                  acceptance_required: true,
                  canonical_body: 'CORPS CGV 2.0',
                },
                {
                  document_type: 'terms_of_service',
                  current_version: '1.0',
                  acceptance_required: false,
                  canonical_body: 'CORPS CGU DEJA COURANT',
                },
              ],
            },
          },
        });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({ data: mockBookings });
      }
      return Promise.reject(new Error('Not found'));
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });
    expect(await screen.findByRole('heading', { name: 'Mise à jour des conditions' })).toBeInTheDocument();
    expect(
      screen.getByText(/Une nouvelle acceptation est requise avant toute nouvelle demande de transport/i)
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', {
        name: /Conditions de réservation et de transport/i,
      })
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole('button', {
        name: /Conditions de réservation et de transport/i,
      })
    );
    expect(screen.getByRole('button', { name: 'PDF' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Imprimer' })).toBeInTheDocument();
    expect(screen.getByText('CORPS CGV 2.0')).toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /Conditions générales d’utilisation — version 1\.0/i })
    ).not.toBeInTheDocument();
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).not.toBeChecked();
    const accept = screen.getByRole('button', { name: 'Accepter les conditions' });
    expect(accept).toBeDisabled();
    const verify = screen.getByRole('button', { name: /Vérifier la demande/i });
    expect(verify).toBeDisabled();

    fireEvent.click(checkbox);
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (String(url).includes('/clients/me/portal-terms-status')) {
        return Promise.resolve({
          data: { data: { status: 'current', documents: [] } },
        });
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({ data: mockBookings });
      }
      return Promise.reject(new Error('Not found'));
    });
    fireEvent.click(accept);

    await waitFor(() => {
      const posts = apiClient.post.mock.calls.filter((call) =>
        String(call[0]).includes('/clients/me/terms-acceptances')
      );
      expect(posts).toHaveLength(1);
      expect(posts[0][1]).toEqual({ accept_current_required_terms: true });
      expect(posts[0][1].terms_version).toBeUndefined();
      expect(posts[0][1].terms_hash).toBeUndefined();
    });
    await waitFor(() => {
      expect(screen.queryByRole('heading', { name: 'Mise à jour des conditions' })).not.toBeInTheDocument();
    });
  });

  it('PORTAL double_validation_v2 — happy path offre → confirmation', async () => {
    const { toast } = require('sonner');
    const pendingBooking = {
      id: 501,
      status: 'pending',
      company_id: null,
      pickup_location: 'Gare',
      dropoff_location: 'HUG',
      amount: 85,
      scheduled_time: toIso(3_600_000),
    };
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url === '/clients/me/contract-flow') {
        return Promise.resolve({ data: { portal_double_validation_enabled: true } });
      }
      if (url === '/clients/me/portal-terms-status') {
        return Promise.resolve({
          data: { requires_reacceptance: false, missing_documents: [] },
        });
      }
      if (String(url).includes('/pending-offer')) {
        return Promise.resolve({
          data: {
            double_validation: true,
            offer: {
              id: 77,
              company_name: 'Trans SA',
              offered_amount: 82,
              maximum_accepted_amount: 95,
              cancellation_policy_text: 'Annulation >24h gratuite',
              offer_content_hash: 'hash-77',
            },
          },
        });
      }
      if (String(url).includes('/bookings')) {
        return Promise.resolve({ data: [pendingBooking] });
      }
      return Promise.resolve({ data: {} });
    });
    apiClient.post.mockResolvedValue({ data: { ok: true } });
    const reloadSpy = jest.fn();
    const originalLocation = window.location;
    delete window.location;
    window.location = { ...originalLocation, reload: reloadSpy };

    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(
      await screen.findByRole('heading', { name: /proposition de transport est disponible/i })
    ).toBeInTheDocument();
    expect(screen.getByText('Trans SA')).toBeInTheDocument();
    expect(screen.getAllByText(/CHF 82\.00/).length).toBeGreaterThan(0);
    expect(screen.getByText(/CHF 95\.00/)).toBeInTheDocument();
    expect(screen.getByText(/Annulation >24h gratuite/)).toBeInTheDocument();

    const confirmBtn = screen.getByRole('button', { name: /Confirmer le transport à CHF 82/i });
    fireEvent.click(confirmBtn);
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      const confirms = apiClient.post.mock.calls.filter((c) =>
        String(c[0]).includes('/confirm-transport')
      );
      expect(confirms).toHaveLength(1);
      expect(confirms[0][1]).toEqual({
        carrier_offer_id: 77,
        offer_content_hash: 'hash-77',
      });
      expect(toast.success).toHaveBeenCalledWith('Votre transport est confirmé.');
    });

    window.location = originalLocation;
  });

  it('PORTAL double_validation_v2 — offre > plafond non présentée', async () => {
    const pendingBooking = {
      id: 502,
      status: 'pending',
      company_id: null,
      pickup_location: 'Gare',
      dropoff_location: 'HUG',
      amount: 85,
      scheduled_time: toIso(3_600_000),
    };
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url === '/clients/me/contract-flow') {
        return Promise.resolve({ data: { portal_double_validation_enabled: true } });
      }
      if (url === '/clients/me/portal-terms-status') {
        return Promise.resolve({
          data: { requires_reacceptance: false, missing_documents: [] },
        });
      }
      if (String(url).includes('/pending-offer')) {
        return Promise.resolve({
          data: {
            double_validation: true,
            offer: {
              id: 88,
              company_name: 'Cher SA',
              offered_amount: 120,
              maximum_accepted_amount: 95,
              cancellation_policy_text: 'x',
              offer_content_hash: 'hash-88',
            },
          },
        });
      }
      if (String(url).includes('/bookings')) {
        return Promise.resolve({ data: [pendingBooking] });
      }
      return Promise.resolve({ data: {} });
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });
    await screen.findByTestId('header-dashboard');
    await waitFor(() => {
      expect(
        screen.queryByRole('heading', { name: /proposition de transport/i })
      ).not.toBeInTheDocument();
    });
  });

  it('PORTAL double_validation_v2 — stale : confirmation refusée, UI non confirmée', async () => {
    const { toast } = require('sonner');
    const pendingBooking = {
      id: 503,
      status: 'pending',
      company_id: null,
      pickup_location: 'Gare',
      dropoff_location: 'HUG',
      amount: 85,
      scheduled_time: toIso(3_600_000),
    };
    apiClient.get.mockImplementation((url) => {
      if (url === '/clients/client-123') {
        return Promise.resolve({ data: mockProfile });
      }
      if (url === '/clients/me/contract-flow') {
        return Promise.resolve({ data: { portal_double_validation_enabled: true } });
      }
      if (url === '/clients/me/portal-terms-status') {
        return Promise.resolve({
          data: { requires_reacceptance: false, missing_documents: [] },
        });
      }
      if (String(url).includes('/pending-offer')) {
        return Promise.resolve({
          data: {
            double_validation: true,
            offer: {
              id: 99,
              company_name: 'Stale SA',
              offered_amount: 80,
              maximum_accepted_amount: 95,
              cancellation_policy_text: 'policy',
              offer_content_hash: 'hash-99',
            },
          },
        });
      }
      if (String(url).includes('/bookings')) {
        return Promise.resolve({ data: [pendingBooking] });
      }
      return Promise.resolve({ data: {} });
    });
    apiClient.post.mockRejectedValue({
      response: { status: 409, data: { error: 'portal_offer_stale', message: 'Offre obsolète' } },
    });

    render(<ClientDashboard />, { wrapper: createWrapper() });
    expect(await screen.findByText('Stale SA')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Confirmer le transport/i }));

    await waitFor(() => {
      expect(toast.success).not.toHaveBeenCalledWith('Votre transport est confirmé.');
    });
  });
});
