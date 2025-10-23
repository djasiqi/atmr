import { renderHook, waitFor } from '@testing-library/react';
import { useDispatchData } from '../useDispatchData';
import * as companyService from '../../services/companyService';

// Mock des services
jest.mock('../../services/companyService');

describe('useDispatchData', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with empty dispatches and loading false', () => {
    const { result } = renderHook(() => useDispatchData('2024-01-15', 'manual'));

    expect(result.current.dispatches).toEqual([]);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should load dispatches for manual mode', async () => {
    const mockData = [
      { id: 1, customer_name: 'Test Client 1', status: 'pending' },
      { id: 2, customer_name: 'Test Client 2', status: 'confirmed' },
    ];

    companyService.fetchCompanyReservations.mockResolvedValue(mockData);

    const { result } = renderHook(() => useDispatchData('2024-01-15', 'manual'));

    // Appeler loadDispatches
    await result.current.loadDispatches();

    await waitFor(() => {
      expect(result.current.dispatches).toEqual(mockData);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    expect(companyService.fetchCompanyReservations).toHaveBeenCalledWith('2024-01-15');
  });

  it('should load assigned dispatches for semi_auto mode', async () => {
    const mockData = [
      { id: 1, customer_name: 'Test Client 1', driver_id: 10, status: 'assigned' },
      { id: 2, customer_name: 'Test Client 2', driver_id: 11, status: 'assigned' },
    ];

    companyService.fetchAssignedReservations.mockResolvedValue(mockData);

    const { result } = renderHook(() => useDispatchData('2024-01-15', 'semi_auto'));

    await result.current.loadDispatches();

    await waitFor(() => {
      expect(result.current.dispatches).toEqual(mockData);
    });

    expect(companyService.fetchAssignedReservations).toHaveBeenCalledWith('2024-01-15');
  });

  it('should handle error during loading', async () => {
    const errorMessage = 'Network error';
    companyService.fetchCompanyReservations.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => useDispatchData('2024-01-15', 'manual'));

    await result.current.loadDispatches();

    await waitFor(() => {
      expect(result.current.error).toBe(errorMessage);
      expect(result.current.dispatches).toEqual([]);
      expect(result.current.loading).toBe(false);
    });
  });

  it('should handle data wrapped in data property', async () => {
    const mockData = {
      data: [
        { id: 1, customer_name: 'Test Client 1' },
        { id: 2, customer_name: 'Test Client 2' },
      ],
    };

    companyService.fetchCompanyReservations.mockResolvedValue(mockData);

    const { result } = renderHook(() => useDispatchData('2024-01-15', 'manual'));

    await result.current.loadDispatches();

    await waitFor(() => {
      expect(result.current.dispatches).toEqual(mockData.data);
    });
  });

  it('should filter returns to confirm in development mode', async () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';

    const mockData = [
      { id: 1, is_return: true, time_confirmed: false, scheduled_time: null },
      { id: 2, is_return: true, time_confirmed: false, scheduled_time: '2024-01-15T10:00:00' },
      { id: 3, is_return: false },
    ];

    companyService.fetchCompanyReservations.mockResolvedValue(mockData);
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

    const { result } = renderHook(() => useDispatchData('2024-01-15', 'manual'));

    await result.current.loadDispatches();

    await waitFor(() => {
      // Vérifier que le log a été appelé avec les retours à confirmer
      const returnsCalls = consoleSpy.mock.calls.filter(
        (call) => call[0] === '[useDispatchData] Retours avec heure à confirmer:'
      );
      expect(returnsCalls.length).toBeGreaterThan(0);
      expect(returnsCalls[0][1]).toBe(2); // 2 retours avec time_confirmed=false
    });

    process.env.NODE_ENV = originalEnv;
    consoleSpy.mockRestore();
  });
});
