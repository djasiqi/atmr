// frontend/tests/hooks/useCompanyData.test.js
import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import useCompanyData from 'hooks/useCompanyData';
import {
  fetchCompanyReservations,
  fetchCompanyDriversCanonical,
  fetchCompanyInfo,
} from 'services/companyService';
import { getAccessToken } from 'hooks/useAuthToken';

// Mocks
jest.mock('services/companyService');
jest.mock('hooks/useAuthToken');
jest.mock('services/companySocket', () => ({
  getCompanySocket: jest.fn(() => null),
  joinCompanyRoom: jest.fn(),
}));
jest.mock('hooks/useCompanySocket', () => ({
  __esModule: true,
  useSocketConnected: jest.fn(() => false),
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }) {
    return React.createElement(QueryClientProvider, { client: queryClient }, children);
  }
  return Wrapper;
}

describe('useCompanyData', () => {
  const mockCompany = {
    id: 1,
    name: 'ATMR Transport',
    email: 'contact@atmr.ch',
    logo_url: 'https://example.com/logo.png',
  };

  const mockReservations = [
    {
      id: 1,
      pickup_location: 'Genève',
      dropoff_location: 'Lausanne',
      status: 'PENDING',
    },
    {
      id: 2,
      pickup_location: 'Vevey',
      dropoff_location: 'Montreux',
      status: 'ASSIGNED',
    },
  ];

  const mockDrivers = [
    {
      id: 1,
      user: { first_name: 'Pierre', last_name: 'Martin' },
      is_available: true,
    },
    {
      id: 2,
      user: { first_name: 'Marie', last_name: 'Dubois' },
      is_available: false,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'log').mockImplementation();
    jest.spyOn(console, 'error').mockImplementation();

    getAccessToken.mockReturnValue('fake-token');
    fetchCompanyInfo.mockResolvedValue(mockCompany);
    fetchCompanyReservations.mockResolvedValue(mockReservations);
    fetchCompanyDriversCanonical.mockResolvedValue(mockDrivers);
  });

  it("devrait charger les données de l'entreprise", async () => {
    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.company).toEqual(mockCompany);
    });

    expect(fetchCompanyInfo).toHaveBeenCalled();
  });

  it('devrait charger les réservations pour un jour spécifique', async () => {
    const { result } = renderHook(() => useCompanyData({ day: '2025-10-16' }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingReservations).toBe(false);
    });

    expect(result.current.reservations).toEqual(mockReservations);
    expect(fetchCompanyReservations).toHaveBeenCalledWith('2025-10-16');
  });

  it('devrait charger les chauffeurs', async () => {
    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingDriver).toBe(false);
    });

    expect(result.current.driver).toEqual(mockDrivers);
    expect(fetchCompanyDriversCanonical).toHaveBeenCalled();
  });

  it('devrait gérer les erreurs de chargement', async () => {
    fetchCompanyReservations.mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingReservations).toBe(false);
    });

    expect(result.current.error).toMatch(/réservations/i);
  });

  it('devrait gérer les erreurs de timeout', async () => {
    const timeoutError = new Error('timeout of 5000ms exceeded');
    timeoutError.code = 'ECONNABORTED';
    fetchCompanyDriversCanonical.mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.error).toBe(
        'La récupération des chauffeurs a pris trop de temps. Veuillez réessayer.'
      );
    });
  });

  it('devrait permettre de recharger les données', async () => {
    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingReservations).toBe(false);
    });

    result.current.reloadReservations();

    await waitFor(() => {
      expect(fetchCompanyReservations).toHaveBeenCalledTimes(2);
    });
  });

  it('ne devrait pas charger company si pas de token', async () => {
    getAccessToken.mockReturnValue(null);
    const ls = window.localStorage;
    const origGet = ls.getItem.bind(ls);
    ls.getItem = jest.fn(() => null);

    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingReservations).toBe(false);
    });

    expect(result.current.company).toBeNull();
    ls.getItem = origGet;
  });

  it('devrait gérer les formats de réponse alternatifs', async () => {
    fetchCompanyReservations.mockResolvedValue({ reservations: mockReservations });
    fetchCompanyDriversCanonical.mockResolvedValue({ driver: mockDrivers });

    const { result } = renderHook(() => useCompanyData(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loadingReservations).toBe(false);
    });

    expect(result.current.reservations).toEqual(mockReservations);
    expect(result.current.driver).toEqual(mockDrivers);
  });
});
