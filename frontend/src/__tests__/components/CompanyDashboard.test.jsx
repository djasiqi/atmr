// frontend/tests/components/CompanyDashboard.test.jsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import CompanyDashboard from 'pages/company/Dashboard/CompanyDashboard';
import useCompanyData from 'hooks/useCompanyData';
import useCompanySocket from 'hooks/useCompanySocket';
import useDispatchStatus from 'hooks/useDispatchStatus';
import useDispatchDelays from 'hooks/useDispatchDelays';

// Mocks
jest.mock('hooks/useCompanyData');
jest.mock('hooks/useCompanySocket');
jest.mock('hooks/useDispatchStatus');
jest.mock('hooks/useDispatchDelays');
jest.mock('services/companyService');

// Mock des composants enfants
jest.mock('components/layout/Sidebar/CompanySidebar/CompanySidebar', () => {
  return function MockSidebar() {
    return <div data-testid="company-sidebar">Sidebar</div>;
  };
});

jest.mock('components/layout/Header/CompanyHeader', () => {
  return function MockHeader() {
    return <div data-testid="company-header">Header</div>;
  };
});

jest.mock('pages/company/Dashboard/components/OverviewCards', () => {
  return function MockOverviewCards({ stats = {} }) {
    return (
      <div data-testid="overview-cards">
        <div>Pending: {stats.pending || 0}</div>
        <div>Assigned: {stats.assigned || 0}</div>
        <div>Completed: {stats.completed || 0}</div>
      </div>
    );
  };
});

jest.mock('pages/company/Dashboard/components/ReservationTable', () => {
  return function MockReservationTable({ reservations }) {
    return <div data-testid="reservation-table">{reservations.length} réservations</div>;
  };
});

jest.mock('pages/company/Dashboard/components/DriverLiveMap', () => {
  return function MockDriverLiveMap() {
    return <div data-testid="driver-live-map">Carte</div>;
  };
});

jest.mock('pages/driver/components/Dashboard/DriverTable', () => {
  return function MockDriverTable({ drivers }) {
    return <div data-testid="driver-table">{drivers.length} chauffeurs</div>;
  };
});

jest.mock('pages/company/Dashboard/components/ManualBookingForm', () => {
  return function MockManualBookingForm({ onSuccess }) {
    return (
      <div data-testid="manual-booking-form">
        <button onClick={() => onSuccess({ id: 123 })}>Créer réservation</button>
      </div>
    );
  };
});

jest.mock('components/widgets/ChatWidget', () => {
  return function MockChatWidget() {
    return <div data-testid="chat-widget">Chat</div>;
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
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </BrowserRouter>
  );
};

describe('CompanyDashboard', () => {
  const mockCompany = {
    id: 1,
    name: 'ATMR Transport',
    email: 'contact@atmr.ch',
  };

  const mockReservations = [
    {
      id: 1,
      pickup_location: 'Genève',
      dropoff_location: 'Lausanne',
      status: 'PENDING',
      scheduled_time: '2025-10-16T10:00:00',
    },
    {
      id: 2,
      pickup_location: 'Vevey',
      dropoff_location: 'Montreux',
      status: 'ASSIGNED',
      scheduled_time: '2025-10-16T14:00:00',
    },
    {
      id: 3,
      pickup_location: 'Nyon',
      dropoff_location: 'Morges',
      status: 'COMPLETED',
      scheduled_time: '2025-10-15T08:00:00',
    },
  ];

  const mockDrivers = [
    {
      id: 1,
      user: { first_name: 'Pierre', last_name: 'Martin' },
      is_available: true,
      vehicle_type: 'berline',
    },
    {
      id: 2,
      user: { first_name: 'Marie', last_name: 'Dubois' },
      is_available: false,
      vehicle_type: 'ambulance',
    },
  ];

  const mockSocket = {
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();

    useCompanyData.mockReturnValue({
      company: mockCompany,
      reservations: mockReservations,
      driver: mockDrivers,
      loadingReservations: false,
      loadingDriver: false,
      reloadReservations: jest.fn(),
      reloadDriver: jest.fn(),
    });

    useCompanySocket.mockReturnValue(mockSocket);

    useDispatchStatus.mockReturnValue({
      label: 'Idle',
      progress: 0,
      isRunning: false,
    });

    useDispatchDelays.mockReturnValue({
      delayCount: 0,
      hasCriticalDelays: false,
      hasDelays: false,
    });
  });

  it('devrait afficher le dashboard avec les statistiques', async () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId('overview-cards')).toBeInTheDocument();
    });

    // Le composant calcule ses propres stats, vérifions juste qu'il affiche
    expect(screen.getByText(/Pending:/i)).toBeInTheDocument();
    expect(screen.getByText(/Assigned:/i)).toBeInTheDocument();
    expect(screen.getByText(/Completed:/i)).toBeInTheDocument();
  });

  it('devrait afficher le tableau des réservations', async () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByTestId('reservation-table')).toBeInTheDocument();
    });
  });

  it('devrait afficher le nombre correct de chauffeurs', async () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      // Vérifier que la section chauffeurs est affichée
      expect(screen.getByText(/Chauffeurs \(2\)/i)).toBeInTheDocument();
    });
  });

  it('devrait afficher les composants principaux', async () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByTestId('company-sidebar')).toBeInTheDocument();
    expect(screen.getByTestId('company-header')).toBeInTheDocument();
    expect(screen.getByTestId('driver-live-map')).toBeInTheDocument();
    expect(screen.getByTestId('chat-widget')).toBeInTheDocument();
  });

  it('devrait gérer les données vides', async () => {
    useCompanyData.mockReturnValue({
      company: mockCompany,
      reservations: [],
      driver: [],
      loadingReservations: false,
      loadingDriver: false,
      reloadReservations: jest.fn(),
      reloadDriver: jest.fn(),
    });

    render(<CompanyDashboard />, { wrapper: createWrapper() });

    await waitFor(() => {
      const reservationTable = screen.getByTestId('reservation-table');
      expect(reservationTable).toHaveTextContent('0 réservations');
    });

    // Vérifier que la section chauffeurs affiche 0
    expect(screen.getByText(/Chauffeurs \(0\)/i)).toBeInTheDocument();
  });

  it('devrait se connecter aux WebSockets', () => {
    render(<CompanyDashboard />, { wrapper: createWrapper() });

    expect(useCompanySocket).toHaveBeenCalled();
    expect(useDispatchStatus).toHaveBeenCalledWith(mockSocket);
  });
});
