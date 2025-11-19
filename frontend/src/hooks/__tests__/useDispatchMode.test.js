import { renderHook, waitFor, act } from '@testing-library/react';
import { useDispatchMode } from '../useDispatchMode';
import apiClient from '../../utils/apiClient';

// Mock apiClient
jest.mock('../../utils/apiClient');

describe('useDispatchMode', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with semi_auto mode by default', () => {
    const { result } = renderHook(() => useDispatchMode());

    expect(result.current.dispatchMode).toBe('semi_auto');
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should load dispatch mode from API', async () => {
    apiClient.get.mockResolvedValue({
      data: { dispatch_mode: 'manual' },
    });

    const { result } = renderHook(() => useDispatchMode());

    await result.current.loadDispatchMode();

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('manual');
      expect(result.current.loading).toBe(false);
    });

    expect(apiClient.get).toHaveBeenCalledWith('/company_dispatch/mode');
  });

  it('should fallback to semi_auto if API returns no mode', async () => {
    apiClient.get.mockResolvedValue({
      data: {},
    });

    const { result } = renderHook(() => useDispatchMode());

    await result.current.loadDispatchMode();

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('semi_auto');
    });
  });

  it('should handle error during loading', async () => {
    const errorMessage = 'Network error';
    apiClient.get.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => useDispatchMode());

    await result.current.loadDispatchMode();

    await waitFor(() => {
      expect(result.current.error).toBe(errorMessage);
      expect(result.current.loading).toBe(false);
    });
  });

  it('should allow manual mode setting', () => {
    const { result } = renderHook(() => useDispatchMode());

    act(() => {
      result.current.setDispatchMode('fully_auto');
    });

    expect(result.current.dispatchMode).toBe('fully_auto');
  });
});
