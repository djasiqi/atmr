// frontend/tests/components/ClientDashboard.test.jsx
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
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

  const openPortalReview = async () => {
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

    fireEvent.click(screen.getByRole('radio', { name: 'Dès que possible' }));
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
    fireEvent.change(screen.getByLabelText(/Date/i), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(screen.getByLabelText(/Heure/i), {
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

    await screen.findByDisplayValue('Rue de Lausanne 1, 1201 Genève');

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
    fireEvent.change(screen.getByLabelText(/Date/i), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(screen.getByLabelText(/Heure/i), {
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
    fireEvent.change(document.getElementById('client-booking-date'), {
      target: { value: `${y}-${m}-${d}` },
    });
    fireEvent.change(document.getElementById('client-booking-time'), {
      target: { value: '10:30' },
    });
  };

  it('affiche une estimation indicative et le titulaire comme débiteur', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });
    await fillOutbound();
    await openPortalReview();

    expect(screen.getAllByText('Jean Dupont').length).toBeGreaterThan(1);
    expect(screen.getByText(/Estimation actuelle : CHF/i)).toBeInTheDocument();
    expect(screen.getByText(/Indicative — le montant final/i)).toBeInTheDocument();
    expect(screen.getByText(/Attribué après confirmation/i)).toBeInTheDocument();
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
    expect(screen.getByRole('button', { name: /Conditions générales de transport 2.0/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Conditions générales d’utilisation 1.0/i })).not.toBeInTheDocument();
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
});
