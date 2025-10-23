import { renderHook, waitFor } from '@testing-library/react';
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
});
