import { renderHook, waitFor, act } from '@testing-library/react';
import { useDispatchMode } from '../useDispatchMode';
import apiClient from '../../utils/apiClient';

jest.mock('../../utils/apiClient');

describe('useDispatchMode', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with null mode and loading true', () => {
    apiClient.get.mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() => useDispatchMode());

    expect(result.current.dispatchMode).toBeNull();
    expect(result.current.loading).toBe(true);
  });

  it('should auto-load dispatch mode from API on mount', async () => {
    apiClient.get.mockResolvedValue({
      data: { dispatch_mode: 'manual' },
    });

    const { result } = renderHook(() => useDispatchMode());

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('manual');
      expect(result.current.loading).toBe(false);
    });

    expect(apiClient.get).toHaveBeenCalledWith('/company_dispatch/mode');
  });

  it('should fallback to manual if API returns no mode', async () => {
    apiClient.get.mockResolvedValue({
      data: {},
    });

    const { result } = renderHook(() => useDispatchMode());

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('manual');
    });
  });

  it('should fallback to manual on error', async () => {
    const errorMessage = 'Network error';
    apiClient.get.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => useDispatchMode());

    await waitFor(() => {
      expect(result.current.error).toBe(errorMessage);
      expect(result.current.dispatchMode).toBe('manual');
      expect(result.current.loading).toBe(false);
    });
  });

  it('should allow manual mode setting', async () => {
    apiClient.get.mockResolvedValue({
      data: { dispatch_mode: 'semi_auto' },
    });

    const { result } = renderHook(() => useDispatchMode());

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('semi_auto');
    });

    act(() => {
      result.current.setDispatchMode('fully_auto');
    });

    expect(result.current.dispatchMode).toBe('fully_auto');
  });

  it('should support reload via loadDispatchMode', async () => {
    apiClient.get
      .mockResolvedValueOnce({ data: { dispatch_mode: 'manual' } })
      .mockResolvedValueOnce({ data: { dispatch_mode: 'semi_auto' } });

    const { result } = renderHook(() => useDispatchMode());

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('manual');
    });

    await act(async () => {
      await result.current.loadDispatchMode();
    });

    await waitFor(() => {
      expect(result.current.dispatchMode).toBe('semi_auto');
    });

    expect(apiClient.get).toHaveBeenCalledTimes(2);
  });
});
