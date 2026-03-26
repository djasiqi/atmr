import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAssignmentActions } from '../useAssignmentActions';
import * as companyService from '../../services/companyService';
import { lirieKeys, listScopeHash } from '../../queryKeys/lirie';

jest.mock('../../services/companyService');

const scopeHash = listScopeHash({ flat: true, include_stats: false });
const day = '2025-10-16';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const assignedKey = lirieKeys.assignedReservations(day);
  const companyResKey = lirieKeys.companyReservations(day, scopeHash);
  queryClient.setQueryData(assignedKey, [
    { id: 1, status: 'pending', driver_id: null },
    { id: 2, status: 'pending', driver_id: null },
  ]);
  queryClient.setQueryData(companyResKey, [
    { id: 1, status: 'pending', driver_id: null },
    { id: 2, status: 'pending', driver_id: null },
  ]);

  const Wrapper = ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  Wrapper.queryClient = queryClient;
  Wrapper.assignedKey = assignedKey;
  Wrapper.companyResKey = companyResKey;
  return Wrapper;
}

describe('useAssignmentActions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with default values', () => {
    const qc = new QueryClient();
    const wrapper = ({ children }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    );
    const { result } = renderHook(() => useAssignmentActions(), { wrapper });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.success).toBeNull();
  });

  it('should assign driver successfully and patch lirie caches', async () => {
    companyService.assignDriver.mockResolvedValue({ success: true });
    const Wrapper = createWrapper();
    const { result } = renderHook(() => useAssignmentActions(), { wrapper: Wrapper });

    const success = await result.current.handleAssignDriver(1, 10);

    await waitFor(() => {
      expect(success).toBe(true);
    });

    expect(companyService.assignDriver).toHaveBeenCalledWith(1, 10);
    const assigned = Wrapper.queryClient.getQueryData(Wrapper.assignedKey);
    expect(assigned.find((r) => r.id === 1).driver_id).toBe(10);
    const company = Wrapper.queryClient.getQueryData(Wrapper.companyResKey);
    expect(company.find((r) => r.id === 1).driver_id).toBe(10);
  });

  it('should handle assignment error and invalidate caches', async () => {
    companyService.assignDriver.mockRejectedValue(new Error('Assignment failed'));
    const Wrapper = createWrapper();
    const invalidateSpy = jest.spyOn(Wrapper.queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useAssignmentActions(), { wrapper: Wrapper });

    const success = await result.current.handleAssignDriver(1, 10);

    await waitFor(() => {
      expect(success).toBe(false);
    });

    expect(result.current.error?.message || String(result.current.error)).toContain('Assignment failed');
    expect(invalidateSpy).toHaveBeenCalled();
  });

  it('should delete reservation successfully', async () => {
    companyService.deleteReservation.mockResolvedValue({ success: true });
    const Wrapper = createWrapper();
    const { result } = renderHook(() => useAssignmentActions(), { wrapper: Wrapper });

    await result.current.handleDeleteReservation(1);

    await waitFor(() => {
      expect(companyService.deleteReservation).toHaveBeenCalledWith(1, null, null);
    });
    const assigned = Wrapper.queryClient.getQueryData(Wrapper.assignedKey);
    expect(assigned.find((r) => r.id === 1)).toBeUndefined();
  });

  it('should handle deletion error', async () => {
    companyService.deleteReservation.mockRejectedValue(new Error('Deletion failed'));
    const Wrapper = createWrapper();

    const { result } = renderHook(() => useAssignmentActions(), { wrapper: Wrapper });

    await expect(result.current.handleDeleteReservation(5)).rejects.toThrow('Deletion failed');

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });
    expect(String(result.current.error?.message || result.current.error)).toContain('Deletion failed');
  });

  it('should set loading state during operations', async () => {
    let resolveAssign;
    const assignPromise = new Promise((resolve) => {
      resolveAssign = resolve;
    });

    companyService.assignDriver.mockImplementation(() => assignPromise);
    const Wrapper = createWrapper();
    const { result } = renderHook(() => useAssignmentActions(), { wrapper: Wrapper });

    result.current.handleAssignDriver(1, 10);

    await waitFor(() => {
      expect(result.current.loading).toBe(true);
    });

    resolveAssign({ success: true });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
