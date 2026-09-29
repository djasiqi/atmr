// frontend/tests/components/ReservationsPage.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ReservationsPage from 'pages/client/Reservations/ReservationsPage';
import { exportBookingsPDF, fetchBookings } from 'services/bookingService';
import { fetchClient } from 'services/clientService';
import { submitContactRequest } from 'services/contactService';
import apiClient from 'utils/apiClient';
import { toast } from 'sonner';

jest.mock('@react-google-maps/api', () => ({
  GoogleMap: ({ children }) => <div data-testid="route-map">{children}</div>,
  Polyline: () => null,
  Marker: () => null,
}));

jest.mock('components/common/GoogleMapsProvider', () => ({
  __esModule: true,
  default: ({ children }) => <>{children}</>,
  useGoogleMapsLoaded: () => ({ isLoaded: false, loadError: null, ensureLoaded: () => {} }),
}));

// Mocks
jest.mock('services/bookingService');
jest.mock('services/clientService');
jest.mock('services/contactService', () => ({
  submitContactRequest: jest.fn(() => Promise.resolve({ trace_id: 'trace' })),
}));
jest.mock('utils/apiClient');
jest.mock('sonner', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    warning: jest.fn(),
    info: jest.fn(),
  },
  Toaster: () => null,
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

// Mock window functions
global.confirm = jest.fn();

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return ({ children }) => (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </BrowserRouter>
  );
};

describe('ReservationsPage', () => {
  const mockClient = {
    id: 42,
    public_id: 'client-123',
    first_name: 'Jean',
    last_name: 'Dupont',
    full_name: 'Jean Dupont',
    user: { email: 'jean.dupont@example.ch', first_name: 'Jean', last_name: 'Dupont' },
  };

  const mockBookings = [
    {
      id: 1,
      pickup_location: 'Genève',
      dropoff_location: 'Lausanne',
      scheduled_time: '2030-12-20T10:00:00',
      status: 'pending',
      amount: 50,
      company_name: 'ATMR Transport',
      driver_name: 'Pierre Martin',
    },
    {
      id: 2,
      pickup_location: 'Vevey',
      dropoff_location: 'Montreux',
      scheduled_time: '2025-10-15T08:00:00',
      status: 'completed',
      amount: 35,
      company_name: 'ATMR Transport',
      driver_name: 'Marie Dubois',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    window.__LIRIE_CLIENT_KPI__ = [];
    localStorage.clear();
    localStorage.setItem('public_id', 'client-123');
    global.confirm.mockReturnValue(true);

    fetchClient.mockResolvedValue(mockClient);
    fetchBookings.mockResolvedValue(mockBookings);
    exportBookingsPDF.mockResolvedValue({});
    apiClient.delete.mockResolvedValue({ status: 200 });
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('devrait afficher la liste des réservations', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Mes courses')).toBeInTheDocument();
    expect(screen.getByTestId('header-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('footer')).toBeInTheDocument();
  });

  it('devrait charger et afficher les réservations du client', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(fetchBookings).toHaveBeenCalledWith('client-123');
    });

    expect(await screen.findByText(/Genève/i)).toBeInTheDocument();
    expect(screen.getByText(/Lausanne/i)).toBeInTheDocument();
  });

  it('devrait séparer les courses à venir et passées', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByRole('heading', { name: 'À venir' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Historique' })).toBeInTheDocument();
  });

  it('range une course terminée dans l’historique même si l’heure prévue est encore à venir', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 46800,
        pickup_location: 'Avenue Ernest-Pictet 9',
        dropoff_location: 'HUG',
        scheduled_time: '2030-12-20T13:15:00',
        status: 'completed',
        amount: 50,
        company_name: 'Emmenez-moi',
      },
    ]);
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByRole('heading', { name: 'Historique' })).toBeInTheDocument();
    expect(screen.queryByText('Prochaine course')).not.toBeInTheDocument();
    expect(screen.getByText(/aucune course à venir/i)).toBeInTheDocument();
    expect(screen.getByText('HUG')).toBeInTheDocument();
  });

  it('détaille le départ et l’arrivée d’une course terminée simple', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 46800,
        pickup_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
        dropoff_location: 'Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève',
        hospital_service: 'Radiologie',
        scheduled_time: '2026-09-30T11:15:00.000Z',
        time_confirmed: true,
        status: 'completed',
        boarded_at: '2026-09-30T11:22:00.000Z',
        completed_at: '2026-09-30T11:41:00.000Z',
        amount: 40,
        company_id: 1,
        pickup_lat: 46.2117,
        pickup_lon: 6.1262,
        dropoff_lat: 46.1936,
        dropoff_lon: 6.1492,
      },
    ]);
    render(<ReservationsPage />, { wrapper: createWrapper() });

    fireEvent.click(await screen.findByRole('button', { name: 'Détails' }));
    expect(await screen.findByText('Trajet')).toBeInTheDocument();
    expect(screen.getByText('Prise en charge')).toBeInTheDocument();
    expect(screen.getByText('Destination')).toBeInTheDocument();
    expect(screen.getByText('Prise en charge 13:22')).toBeInTheDocument();
    expect(screen.getByText('Dépôt 13:41')).toBeInTheDocument();
    expect(screen.queryByText(/Départ/)).not.toBeInTheDocument();
    expect(screen.getByText(/Avenue Ernest-Pictet 9/)).toBeInTheDocument();
    expect(screen.getByText(/Radiologie/)).toBeInTheDocument();
    expect(screen.getByLabelText('Aperçu de l’itinéraire')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Télécharger la facture' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Signaler un événement' })).toBeInTheDocument();
  });

  it('envoie à Lirie un signalement lié à la course terminée', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 46800,
        pickup_location: 'Avenue Ernest-Pictet 9, 1203, Genève',
        dropoff_location: 'Hôpitaux Universitaires de Genève',
        scheduled_time: '2026-09-30T11:15:00.000Z',
        status: 'completed',
        amount: 40,
      },
    ]);
    render(<ReservationsPage />, { wrapper: createWrapper() });

    fireEvent.click(await screen.findByRole('button', { name: 'Détails' }));
    fireEvent.click(screen.getByRole('button', { name: 'Signaler un événement' }));
    expect(screen.getByText(/info@lirie\.ch/)).toBeInTheDocument();
    expect(screen.getByText(/Statut : Terminée/)).toBeInTheDocument();
    expect(screen.getByText(/Référence : #46800/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Que s’est-il passé ?'), {
      target: { value: 'Le chauffeur est arrivé en retard.' },
    });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer à Lirie' }));

    await waitFor(() => {
      expect(submitContactRequest).toHaveBeenCalledTimes(1);
    });
    const payload = submitContactRequest.mock.calls[0][0];
    expect(payload.category).toBe('support');
    expect(payload.reference).toBe('#46800');
    expect(payload.email).toBe('jean.dupont@example.ch');
    expect(payload.subject_detail).toBe('delay');
    expect(payload.message).toContain('Le chauffeur est arrivé en retard.');
    expect(payload.message).toContain('Avenue Ernest-Pictet 9');
    expect(payload.message).toContain('Hôpitaux Universitaires de Genève');
    expect(payload.message).toContain('Statut : Terminée');
    expect(payload.message).toContain('Montant : 40.00 CHF');
    expect(payload.message).toContain('Référence : #46800');
  });

  it('détaille les étapes d’une course terminée à plusieurs trajets', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 46797,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 1,
        is_return: false,
        status: 'completed',
        amount: 40,
        scheduled_time: '2026-09-29T20:50:00.000Z',
        pickup_location: 'Avenue Ernest-Pictet 9',
        dropoff_location: 'HUG',
        company_name: 'Emmenez-moi',
      },
      {
        id: 46798,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 2,
        is_return: false,
        status: 'completed',
        amount: 40,
        scheduled_time: '2026-09-29T21:45:00.000Z',
        pickup_location: 'HUG',
        dropoff_location: 'Clinique de Joli-Mont',
        company_name: 'Emmenez-moi',
      },
      {
        id: 46799,
        company_id: 1,
        route_group_id: 'grp',
        route_sequence_number: 3,
        is_return: true,
        parent_booking_id: 46798,
        status: 'completed',
        amount: 40,
        scheduled_time: null,
        pickup_location: 'Clinique de Joli-Mont',
        dropoff_location: 'Avenue Ernest-Pictet 9',
        company_name: 'Emmenez-moi',
      },
    ]);
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText(/3 trajets/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Détails' }));
    expect(await screen.findByText('Étape 1')).toBeInTheDocument();
    expect(screen.getByText('Étape 2')).toBeInTheDocument();
    expect(screen.getByText('Retour')).toBeInTheDocument();
    expect(screen.getByText('Clinique de Joli-Mont')).toBeInTheDocument();
    expect(screen.queryByText('Progression du trajet')).not.toBeInTheDocument();
  });

  it('affiche la section Prochaine course lorsqu’une course future existe', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Prochaine course')).toBeInTheDocument();
    expect(screen.getByText('Aucune autre course programmée.')).toBeInTheDocument();
  });

  it('devrait filtrer par statut', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    const filterToutes = await screen.findByRole('button', { name: 'Toutes' });
    expect(filterToutes).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Terminées' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Terminées' })).toHaveAttribute('aria-pressed', 'true');
    });
  });

  it('devrait trier par date', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    const sortSelect = await screen.findByDisplayValue('Par date');
    expect(sortSelect).toBeInTheDocument();

    fireEvent.change(sortSelect, { target: { value: 'amount' } });

    await waitFor(() => {
      expect(sortSelect.value).toBe('amount');
    });
  });

  it("devrait permettre d'annuler une réservation", async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    // Attendre que les réservations soient chargées
    const cancelButtons = await screen.findAllByText('Annuler', {}, { timeout: 3000 });
    expect(cancelButtons.length).toBeGreaterThan(0);

    fireEvent.click(cancelButtons[0]);

    await waitFor(() => {
      expect(global.confirm).toHaveBeenCalledWith(
        'Confirmer l’annulation de cette réservation ? Les conditions de remboursement (délais 24 h / 4 h, aller-retour) sont rappelées dans la section « Modification et annulation » au-dessus des boutons.'
      );
    });

    expect(apiClient.delete).toHaveBeenCalledWith('/bookings/1');
    await waitFor(() => {
      expect(fetchBookings).toHaveBeenCalledTimes(2);
    });
    expect(toast.success).toHaveBeenCalledWith('Réservation annulée.');
  });

  it("ne devrait pas annuler si l'utilisateur refuse", async () => {
    global.confirm.mockReturnValue(false);
    render(<ReservationsPage />, { wrapper: createWrapper() });

    // Attendre que les réservations soient chargées
    const cancelButtons = await screen.findAllByText('Annuler', {}, { timeout: 3000 });
    expect(cancelButtons.length).toBeGreaterThan(0);

    fireEvent.click(cancelButtons[0]);

    await waitFor(() => {
      expect(global.confirm).toHaveBeenCalled();
    });

    expect(apiClient.delete).not.toHaveBeenCalled();
  });

  it('affiche le prix maximum tant que le chauffeur n’a pas confirmé', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 11,
        route_group_id: 'grp',
        route_sequence_number: 1,
        pickup_location: 'Domicile',
        dropoff_location: 'HUG',
        scheduled_time: '2030-12-20T21:00:00',
        status: 'pending',
        amount: 25,
        is_return: false,
        client: { client_type: 'PORTAL' },
        client_type: 'PORTAL',
      },
      {
        id: 12,
        route_group_id: 'grp',
        route_sequence_number: 2,
        pickup_location: 'HUG',
        dropoff_location: 'Clinique',
        scheduled_time: '2030-12-20T21:45:00',
        status: 'pending',
        amount: 0.5,
        is_return: false,
        time_confirmed: true,
        client: { client_type: 'PORTAL' },
      },
      {
        id: 13,
        route_group_id: 'grp',
        route_sequence_number: 3,
        pickup_location: 'Clinique',
        dropoff_location: 'Domicile',
        scheduled_time: null,
        status: 'pending',
        amount: 25,
        is_return: true,
        time_confirmed: false,
        client: { client_type: 'PORTAL' },
      },
    ]);

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Prix maximum autorisé')).toBeInTheDocument();
    expect(screen.getByText('135.00 CHF')).toBeInTheDocument();
    expect(
      screen.getByText(/En attente de confirmation de l’entreprise de transport/)
    ).toBeInTheDocument();
    expect(screen.queryByText(/50\.00 CHF/)).not.toBeInTheDocument();
    expect(screen.queryByText('(3 transports)')).not.toBeInTheDocument();
  });

  it('n’affiche pas un paiement effectué pour une demande portail', async () => {
    fetchBookings.mockResolvedValue([
      {
        id: 9,
        pickup_location: 'Genève',
        dropoff_location: 'Lausanne',
        scheduled_time: '2030-12-20T10:00:00',
        status: 'pending',
        amount: 45,
        client: { client_type: 'PORTAL' },
        client_type: 'PORTAL',
      },
    ]);

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Demande transmise aux entreprises')).toBeInTheDocument();
    expect(screen.queryByText('Paiement requis')).not.toBeInTheDocument();
  });

  it('devrait afficher un message si aucune réservation', async () => {
    fetchBookings.mockResolvedValue([]);

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText(/Vous n'avez aucune course à venir/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Réserver une course/i })).toBeInTheDocument();
    expect(screen.getByText('Aucune course passée.')).toBeInTheDocument();
  });

  it('devrait gérer les erreurs de chargement', async () => {
    fetchBookings.mockRejectedValue(new Error('Network error'));

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(
      await screen.findByText('Impossible de charger les réservations.')
    ).toBeInTheDocument();
  });

  it("émet l'événement KPI d'export historique", async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });
    await screen.findByText('Historique');
    fireEvent.click(await screen.findByRole('button', { name: /Exporter en PDF/i }));
    await waitFor(() => {
      expect(window.__LIRIE_CLIENT_KPI__.some((e) => e.name === 'history_export_clicked')).toBe(
        true
      );
    });
  });
});
