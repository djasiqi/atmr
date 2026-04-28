// frontend/tests/hooks/useDriver.test.js
import React from 'react';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import useDriver from 'hooks/useDriver';
import {
  fetchCompanyDriversCanonical,
  updateDriverStatus,
  deleteDriver,
} from 'services/companyService';
import { getCompanySocket, joinCompanyRoom } from 'services/companySocket';
import { useLirieCompany } from 'hooks/useLirieCompany';
import { useSocketConnected } from 'hooks/useCompanySocket';
import { lirieKeys } from '../../queryKeys/lirie';

function createTestClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

function withQuery(client) {
  // eslint-disable-next-line react/prop-types
  return function Wrapper({ children }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

jest.mock('hooks/useLirieCompany');
jest.mock('hooks/useCompanySocket');
jest.mock('services/companyService');
jest.mock('services/companySocket');

describe('useDriver', () => {
  let handlers;
  let socket;

  const mockDrivers = [
    {
      id: 1,
      company_id: 99,
      first_name: 'Pierre',
      last_name: 'Martin',
      status: 'available',
      location_status: 'offline',
      presence_status: 'offline',
      last_seen_seconds: 1200,
      is_available: true,
      is_active: true,
      vehicle_type: 'berline',
    },
    {
      id: 2,
      company_id: 99,
      first_name: 'Marie',
      last_name: 'Dubois',
      is_available: false,
      is_active: true,
      vehicle_type: 'ambulance',
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useRealTimers();
    jest.spyOn(console, 'error').mockImplementation();
    useLirieCompany.mockReturnValue({ company: { id: 99 } });
    useSocketConnected.mockReturnValue(false);

    handlers = {};
    socket = {
      on: jest.fn((event, cb) => {
        handlers[event] = cb;
      }),
      off: jest.fn((event) => {
        delete handlers[event];
      }),
    };
    getCompanySocket.mockReturnValue(socket);
    joinCompanyRoom.mockResolvedValue();
    fetchCompanyDriversCanonical.mockResolvedValue(mockDrivers);
    updateDriverStatus.mockResolvedValue({ success: true });
    deleteDriver.mockResolvedValue({ success: true });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('devrait charger les chauffeurs au montage', async () => {
    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toEqual(mockDrivers);
    expect(fetchCompanyDriversCanonical).toHaveBeenCalled();
    expect(joinCompanyRoom).toHaveBeenCalledWith(99);
  });

  it("devrait mettre à jour le statut d'un chauffeur", async () => {
    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.toggleDriverStatus(1, false);
    });

    expect(updateDriverStatus).toHaveBeenCalledWith(1, false);
    await waitFor(() => {
      const fromQ = client.getQueryData(lirieKeys.companyDrivers());
      const row = (fromQ || []).find((x) => Number(x.id) === 1);
      expect(row?.is_active).toBe(false);
    });
  });

  it('devrait supprimer un chauffeur', async () => {
    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toHaveLength(2);

    await act(async () => {
      await result.current.deleteDriverById(2);
    });

    expect(deleteDriver).toHaveBeenCalledWith(2);
    await waitFor(() => {
      const fromQ = client.getQueryData(lirieKeys.companyDrivers()) || [];
      expect(fromQ).toHaveLength(1);
      expect(fromQ.find((d) => Number(d.id) === 2)).toBeUndefined();
    });
  });

  it('devrait gérer les erreurs de chargement', async () => {
    fetchCompanyDriversCanonical.mockRejectedValue(new Error('Network error'));

    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Erreur lors du chargement des chauffeurs.');
  });

  it('devrait permettre de rafraîchir les chauffeurs', async () => {
    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(fetchCompanyDriversCanonical).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.refreshDrivers();
    });

    expect(fetchCompanyDriversCanonical).toHaveBeenCalledTimes(2);
  });

  it('devrait gérer un tableau vide de chauffeurs', async () => {
    fetchCompanyDriversCanonical.mockResolvedValue([]);

    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.drivers).toEqual([]);
  });

  it('devrait gérer les erreurs de mise à jour de statut', async () => {
    updateDriverStatus.mockRejectedValue(new Error('Update failed'));

    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.toggleDriverStatus(1, false);
    });

    expect(console.error).toHaveBeenCalledWith(
      'Erreur lors de la mise à jour du statut :',
      expect.any(Error)
    );
  });

  it('devrait gérer les erreurs de suppression', async () => {
    deleteDriver.mockRejectedValue(new Error('Delete failed'));

    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    await act(async () => {
      await result.current.deleteDriverById(1);
    });

    expect(console.error).toHaveBeenCalledWith(
      'Erreur lors de la suppression :',
      expect.any(Error)
    );

    expect(result.current.drivers).toHaveLength(2);
  });

  it('applique les deltas socket (cache TanStack via overlay)', async () => {
    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      handlers.driver_live_state_update({
        driver_id: 1,
        lat: 46.2044,
        lng: 6.1432,
        status: 'busy',
        location_status: 'live',
        presence_status: 'online',
        last_seen_seconds: 3,
        event_id: 90101,
      });
    });

    await waitFor(() => {
      const updated = result.current.drivers.find((d) => d.id === 1);
      expect(updated.status).toBe('busy');
    });
    const updated = result.current.drivers.find((d) => d.id === 1);
    expect(updated.location_status).toBe('live');
    expect(updated.presence_status).toBe('online');
    expect(updated.last_seen_seconds).toBe(3);
    expect(updated.latitude).toBe(46.2044);
    expect(updated.longitude).toBe(6.1432);
  });

  it("au reconnect, resynchronise si le snapshot a plus d'1 min (overlay)", async () => {
    const t0 = new Date('2025-01-10T10:00:00.000Z');
    const t1 = new Date('2025-01-10T10:02:00.000Z');
    jest.useFakeTimers();
    jest.setSystemTime(t0);

    const client = createTestClient();
    const { result } = renderHook(() => useDriver(), { wrapper: withQuery(client) });
    await waitFor(() => expect(result.current.loading).toBe(false));
    const before = fetchCompanyDriversCanonical.mock.calls.length;

    act(() => {
      jest.setSystemTime(t1);
    });
    act(() => {
      window.dispatchEvent(new CustomEvent('company_socket_reconnected'));
    });

    await waitFor(
      () => {
        expect(fetchCompanyDriversCanonical.mock.calls.length).toBeGreaterThan(before);
      },
      { timeout: 5000 }
    );
    expect(result.current.error).toBeNull();
    jest.useRealTimers();
  });
});
