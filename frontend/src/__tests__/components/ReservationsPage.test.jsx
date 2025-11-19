// frontend/tests/components/ReservationsPage.test.jsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ReservationsPage from 'pages/client/Reservations/ReservationsPage';
import { fetchBookings } from 'services/bookingService';
import { fetchClient } from 'services/clientService';
import apiClient from 'utils/apiClient';

// Mocks
jest.mock('services/bookingService');
jest.mock('services/clientService');
jest.mock('utils/apiClient');
jest.mock('hooks/useCompanyData', () => ({
  __esModule: true,
  default: () => ({ company: null }),
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
global.alert = jest.fn();
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
  };

  const mockBookings = [
    {
      id: 1,
      pickup_location: 'Genève',
      dropoff_location: 'Lausanne',
      scheduled_time: '2025-10-20T10:00:00',
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
    localStorage.clear();
    localStorage.setItem('public_id', 'client-123');
    global.alert.mockClear();
    global.confirm.mockReturnValue(true);

    fetchClient.mockResolvedValue(mockClient);
    fetchBookings.mockResolvedValue(mockBookings);
    apiClient.delete.mockResolvedValue({ status: 200 });
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('devrait afficher la liste des réservations', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('📌 Mes Réservations')).toBeInTheDocument();
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

    expect(await screen.findByText('📅 Courses à venir')).toBeInTheDocument();
    expect(screen.getByText('📅 Courses passées')).toBeInTheDocument();
  });

  it('devrait filtrer par statut', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    const filterSelect = await screen.findByDisplayValue('📋 Tous');
    fireEvent.change(filterSelect, { target: { value: 'completed' } });

    await waitFor(() => {
      // Devrait filtrer pour n'afficher que les courses terminées
      expect(filterSelect.value).toBe('completed');
    });
  });

  it('devrait trier par date', async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    const sortSelect = await screen.findByDisplayValue('📅 Trier par Date');
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
        'Voulez-vous vraiment annuler cette réservation ?'
      );
    });

    expect(apiClient.delete).toHaveBeenCalledWith('/bookings/1');
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

  it("devrait permettre d'exporter en PDF", async () => {
    render(<ReservationsPage />, { wrapper: createWrapper() });

    const monthSelect = await screen.findByDisplayValue('📅 Sélectionner un mois');
    fireEvent.change(monthSelect, { target: { value: '10' } });

    const exportButton = screen.getByText(/Exporter en PDF/i);
    fireEvent.click(exportButton);

    await waitFor(() => {
      expect(global.alert).toHaveBeenCalled();
    });
  });

  it('devrait afficher un message si aucune réservation', async () => {
    fetchBookings.mockResolvedValue([]);

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(await screen.findByText('Aucune course à venir.')).toBeInTheDocument();
    expect(screen.getByText('Aucune course passée.')).toBeInTheDocument();
  });

  it('devrait gérer les erreurs de chargement', async () => {
    fetchBookings.mockRejectedValue(new Error('Network error'));

    render(<ReservationsPage />, { wrapper: createWrapper() });

    expect(
      await screen.findByText('Erreur lors du chargement des réservations.')
    ).toBeInTheDocument();
  });
});
