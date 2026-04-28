// frontend/tests/components/ClientDashboard.test.jsx
import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ClientDashboard from 'pages/client/Dashboard/ClientDashboard';
import apiClient from 'utils/apiClient';
import { startSaferpayHostedCheckout } from 'services/clientSaferpayPaymentService';

const mockNavigate = jest.fn();

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
    user: {
      first_name: 'Jean',
      last_name: 'Dupont',
      email: 'jean.dupont@example.com',
    },
    billing_address: 'Rue de Lausanne 1, 1201 Genève',
  };

  const mockBookings = [];

  beforeEach(() => {
    mockNavigate.mockClear();
    jest.clearAllMocks();
    window.__LIRIE_CLIENT_KPI__ = [];
    localStorage.clear();
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
    apiClient.post.mockResolvedValue({
      data: {
        data: {
          booking_id: 999,
          trace_id: 'trace',
          booking: {
            id: 999,
            amount: 50,
            billed_to_type: 'patient',
            status: 'awaiting_client_payment',
            pickup_location: 'A',
            dropoff_location: 'B',
          },
        },
      },
    });
  });

  afterEach(() => {
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
    fireEvent.click(screen.getByRole('button', { name: /Valider la demande de transport/i }));

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
    expect(screen.getByText('Aller-retour')).toBeInTheDocument();
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
    expect(screen.getByText('Retour')).toBeInTheDocument();
    expect(screen.queryByText('Aller-retour')).not.toBeInTheDocument();
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
    expect(screen.getByText('Aller-retour')).toBeInTheDocument();
  });

  it('désactive date et heure quand dès que possible est activé', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    const dateInput = await screen.findByLabelText(/Date/i);
    const timeInput = screen.getByLabelText(/Heure/i);
    const asapRadio = screen.getByRole('radio', { name: /Dès que possible/i });

    expect(dateInput).not.toBeDisabled();
    expect(timeInput).not.toBeDisabled();

    fireEvent.click(asapRadio);

    expect(dateInput).toBeDisabled();
    expect(timeInput).toBeDisabled();
    expect(asapRadio).toBeChecked();
  });

  it('la réservation reste possible si estimation itinéraire échoue', async () => {
    jest.useFakeTimers();
    apiClient.post.mockImplementation((url) => {
      if (url === '/ai/optimized-route') {
        return Promise.reject(new Error('route-failed'));
      }
      if (url.includes('/bookings')) {
        return Promise.resolve({
          data: {
            data: {
              booking_id: 777,
              trace_id: 't',
              booking: {
                id: 777,
                pickup_location: 'Genève',
                dropoff_location: 'Lausanne',
                scheduled_time: toIso(2 * 60 * 60 * 1000),
                status: 'awaiting_client_payment',
                amount: 50,
                billed_to_type: 'patient',
              },
            },
          },
        });
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

    await act(async () => {
      jest.advanceTimersByTime(2100);
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Impossible d’estimer ce trajet pour le moment/i)
      ).toBeInTheDocument();
    });

    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const y = tomorrow.getFullYear();
    const m = String(tomorrow.getMonth() + 1).padStart(2, '0');
    const d = String(tomorrow.getDate()).padStart(2, '0');
    fireEvent.change(screen.getByLabelText(/Date/i), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(screen.getByLabelText(/Heure/i), {
      target: { value: '10:30' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Valider la demande de transport/i }));

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
    jest.useRealTimers();
  });

  it('lance Saferpay après réservation lorsque billed_to_type est patient', async () => {
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
    fireEvent.change(screen.getByLabelText(/Date/i), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(screen.getByLabelText(/Heure/i), {
      target: { value: '10:30' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Valider la demande de transport/i }));

    await waitFor(() => {
      expect(startSaferpayHostedCheckout).toHaveBeenCalledWith(999);
      expect(mockNavigate).not.toHaveBeenCalledWith(
        expect.stringContaining('/client/payment/saferpay/start'),
        expect.anything()
      );
    });
  });

  it('ne lance pas Saferpay pour une réservation tiers payeur (assurance)', async () => {
    startSaferpayHostedCheckout.mockClear();
    apiClient.post.mockResolvedValue({
      data: {
        data: {
          booking_id: 1002,
          trace_id: 'trace',
          booking: {
            id: 1002,
            amount: 50,
            billed_to_type: 'insurance',
            status: 'pending',
          },
        },
      },
    });

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
    fireEvent.change(screen.getByLabelText(/Date/i), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(screen.getByLabelText(/Heure/i), {
      target: { value: '10:30' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Valider la demande de transport/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/clients/client-123/bookings',
        expect.any(Object),
        expect.any(Object)
      );
    });

    expect(startSaferpayHostedCheckout).not.toHaveBeenCalled();
  });
});
