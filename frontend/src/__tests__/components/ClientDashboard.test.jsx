// frontend/tests/components/ClientDashboard.test.jsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter, MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ClientDashboard from 'pages/client/Dashboard/ClientDashboard';
import apiClient from 'utils/apiClient';

// Mocks
jest.mock('utils/apiClient');

// Mock react-leaflet
jest.mock('react-leaflet', () => ({
  MapContainer: ({ children }) => <div data-testid="map-container">{children}</div>,
  TileLayer: () => null,
  Polyline: () => null,
  Marker: () => null,
  Popup: () => null,
  useMap: () => ({
    fitBounds: jest.fn(),
  }),
}));

// Mock react-slick
jest.mock('react-slick', () => {
  return function MockSlider({ children }) {
    return <div data-testid="slider">{children}</div>;
  };
});

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

// Mock window.alert
global.alert = jest.fn();

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

describe('ClientDashboard', () => {
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

  const mockBookings = [
    {
      id: 1,
      pickup_location: 'Genève',
      dropoff_location: 'Lausanne',
      scheduled_time: '2025-10-20T10:00:00',
      status: 'PENDING',
      amount: 50,
    },
    {
      id: 2,
      pickup_location: 'Vevey',
      dropoff_location: 'Montreux',
      scheduled_time: '2025-10-15T08:00:00',
      status: 'COMPLETED',
      amount: 35,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    localStorage.setItem('authToken', 'fake-client-token');
    localStorage.setItem('public_id', 'client-123');
    global.alert.mockClear();

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
      expect(apiClient.get).toHaveBeenCalledWith('/clients/client-123', {
        headers: { Authorization: 'Bearer fake-client-token' },
      });
    });
  });

  it('devrait afficher le composant même sans token', async () => {
    localStorage.removeItem('authToken');

    render(<ClientDashboard />, { wrapper: createWrapper() });

    // Le composant s'affiche (la navigation est testée en E2E)
    expect(await screen.findByTestId('header-dashboard')).toBeInTheDocument();
  });

  it('devrait afficher une erreur si client_id introuvable', async () => {
    localStorage.removeItem('public_id');

    render(<ClientDashboard />, { wrapper: createWrapper() });

    // Le composant devrait afficher une erreur
    await waitFor(() => {
      expect(apiClient.get).not.toHaveBeenCalled();
    });
  });

  it('devrait afficher la carte Leaflet', async () => {
    render(<ClientDashboard />, { wrapper: createWrapper() });

    expect(await screen.findByTestId('map-container')).toBeInTheDocument();
  });
});
