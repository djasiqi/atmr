import { renderHook, waitFor, act } from '@testing-library/react';
import { useLiveDelays } from '../useLiveDelays';
import * as dispatchMonitoringService from '../../services/dispatchMonitoringService';

// Mock des services
jest.mock('../../services/dispatchMonitoringService');

describe('useLiveDelays', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with empty delays and summary', () => {
    const { result } = renderHook(() => useLiveDelays('2024-01-15'));

    expect(result.current.delays).toEqual([]);
    expect(result.current.summary).toBeNull();
  });

  it('should load delays and summary', async () => {
    const mockResponse = {
      delays: [
        { booking_id: 1, delay_minutes: 15, reason: 'Traffic' },
        { booking_id: 2, delay_minutes: 30, reason: 'Accident' },
      ],
      summary: {
        total_delays: 2,
        avg_delay_minutes: 22.5,
        critical_delays: 1,
      },
    };

    dispatchMonitoringService.getLiveDelays.mockResolvedValue(mockResponse);

    const { result } = renderHook(() => useLiveDelays('2024-01-15'));

    await result.current.loadDelays();

    await waitFor(() => {
      expect(result.current.delays).toEqual(mockResponse.delays);
      expect(result.current.summary).toEqual(mockResponse.summary);
    });

    expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledWith('2024-01-15');
  });

  it('should handle error during loading', async () => {
    dispatchMonitoringService.getLiveDelays.mockRejectedValue(new Error('API Error'));

    const { result } = renderHook(() => useLiveDelays('2024-01-15'));

    await result.current.loadDelays();

    await waitFor(() => {
      expect(result.current.delays).toEqual([]);
      expect(result.current.summary).toBeNull();
    });
  });

  it('should reload delays when date changes', async () => {
    dispatchMonitoringService.getLiveDelays.mockResolvedValue({
      delays: [],
      summary: null,
    });

    const { result, rerender } = renderHook(({ date }) => useLiveDelays(date), {
      initialProps: { date: '2024-01-15' },
    });

    // Appeler loadDelays pour la première date
    await result.current.loadDelays();

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledWith('2024-01-15');
    });

    // Changer la date et recharger
    rerender({ date: '2024-01-16' });
    await result.current.loadDelays();

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledWith('2024-01-16');
    });
  });

  it('coalesces scheduleLoadDelays within MIN interval after a successful GET (P3)', async () => {
    dispatchMonitoringService.getLiveDelays.mockResolvedValue({
      delays: [],
      summary: null,
    });
    const mockSocket = { on: jest.fn(), off: jest.fn() };

    jest.useFakeTimers();
    const { result } = renderHook(() =>
      useLiveDelays('2024-01-15', true, { socket: mockSocket })
    );

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalled();
    });
    expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledTimes(1);

    dispatchMonitoringService.getLiveDelays.mockClear();

    act(() => {
      result.current.scheduleLoadDelays();
    });
    act(() => {
      jest.advanceTimersByTime(400);
    });

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).not.toHaveBeenCalled();
    });

    act(() => {
      jest.advanceTimersByTime(15000);
    });
    act(() => {
      result.current.scheduleLoadDelays();
    });
    act(() => {
      jest.advanceTimersByTime(400);
    });

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledTimes(1);
    });

    jest.useRealTimers();
  });

  it('debounces scheduleLoadDelays into a single GET (P3)', async () => {
    dispatchMonitoringService.getLiveDelays.mockResolvedValue({
      delays: [],
      summary: null,
    });
    const mockSocket = { on: jest.fn(), off: jest.fn() };

    jest.useFakeTimers();
    const { result } = renderHook(() =>
      useLiveDelays('2024-01-15', true, { socket: mockSocket })
    );

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalled();
    });

    dispatchMonitoringService.getLiveDelays.mockClear();

    // Hors fenêtre de coalescing (MIN_GET_INTERVAL_MS), sinon scheduleLoadDelays est ignoré.
    act(() => {
      jest.advanceTimersByTime(15000);
    });

    act(() => {
      result.current.scheduleLoadDelays();
      result.current.scheduleLoadDelays();
      result.current.scheduleLoadDelays();
    });

    act(() => {
      jest.advanceTimersByTime(399);
    });
    expect(dispatchMonitoringService.getLiveDelays).not.toHaveBeenCalled();

    act(() => {
      jest.advanceTimersByTime(1);
    });

    jest.useRealTimers();

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledTimes(1);
    });
    expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledWith('2024-01-15');
  });

  it('serializes overlapping loadDelays (single-flight + pending refresh, P3)', async () => {
    let resolveFirst;
    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    dispatchMonitoringService.getLiveDelays
      .mockImplementationOnce(() => firstPromise)
      .mockResolvedValue({ delays: [], summary: null });

    const { result } = renderHook(() => useLiveDelays('2024-01-15', false));

    let p1;
    let p2;
    await act(async () => {
      p1 = result.current.loadDelays();
      p2 = result.current.loadDelays();
    });

    expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveFirst({ delays: [], summary: null });
      await p1;
      await p2;
    });

    await waitFor(() => {
      expect(dispatchMonitoringService.getLiveDelays).toHaveBeenCalledTimes(2);
    });
  });
});
