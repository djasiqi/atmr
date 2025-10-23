// frontend/tests/hooks/useDispatchDelays.test.js
import { renderHook, waitFor } from '@testing-library/react';
import { useDispatchDelays } from 'hooks/useDispatchDelays';
import { getLiveDelays, getOptimizerStatus } from 'services/dispatchMonitoringService';

// Mocks
jest.mock('services/dispatchMonitoringService');

// Mock timers
jest.useFakeTimers();

describe('useDispatchDelays', () => {
  const mockDelays = [
    {
      booking_id: 101,
      delay_minutes: 15,
      severity: 'warning',
      driver_name: 'Pierre Martin',
      suggestions: [
        {
          type: 'reassign',
          priority: 'medium',
          message: 'Réassigner au chauffeur disponible',
        },
      ],
    },
    {
      booking_id: 102,
      delay_minutes: 45,
      severity: 'critical',
      driver_name: 'Marie Dubois',
      suggestions: [
        {
          type: 'urgent',
          priority: 'critical',
          message: 'Contact client immédiat',
        },
      ],
    },
  ];

  const mockSummary = {
    total: 10,
    on_time: 7,
    late: 2,
    early: 1,
    average_delay: 12.5,
  };

  const mockOptimizerStatus = {
    running: true,
    interval: 120,
    last_run: '2025-10-16T10:00:00',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation();

    getLiveDelays.mockResolvedValue({
      delays: mockDelays,
      summary: mockSummary,
    });

    getOptimizerStatus.mockResolvedValue(mockOptimizerStatus);
  });

  afterEach(() => {
    jest.clearAllTimers();
  });

  it('devrait charger les retards au montage', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.delays).toEqual(mockDelays);
    expect(result.current.summary).toEqual(mockSummary);
    expect(getLiveDelays).toHaveBeenCalledWith('2025-10-16');
  });

  it('devrait utiliser la date du jour si non spécifiée', async () => {
    const today = new Date().toISOString().split('T')[0];

    const { result } = renderHook(() => useDispatchDelays());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(getLiveDelays).toHaveBeenCalledWith(today);
  });

  it('devrait calculer les compteurs correctement', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.delayCount).toBe(2);
    expect(result.current.onTimeCount).toBe(7);
    expect(result.current.earlyCount).toBe(1);
    expect(result.current.totalCount).toBe(10);
    expect(result.current.averageDelay).toBe(12.5);
  });

  it('devrait détecter les retards critiques', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.hasDelays).toBe(true);
    expect(result.current.hasCriticalDelays).toBe(true);
  });

  it("devrait charger le statut de l'optimiseur", async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.optimizerStatus).toEqual(mockOptimizerStatus);
    });

    expect(getOptimizerStatus).toHaveBeenCalled();
  });

  it('devrait rafraîchir automatiquement avec interval', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16', 5000));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Premier appel initial
    expect(getLiveDelays).toHaveBeenCalledTimes(1);

    // Avancer le timer de 5 secondes
    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(getLiveDelays).toHaveBeenCalledTimes(2);
    });

    // Avancer encore 5 secondes
    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(getLiveDelays).toHaveBeenCalledTimes(3);
    });
  });

  it('ne devrait pas auto-refresh si interval = 0', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16', 0));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Premier appel
    expect(getLiveDelays).toHaveBeenCalledTimes(1);

    // Avancer le temps
    jest.advanceTimersByTime(10000);

    // Pas d'appel supplémentaire
    expect(getLiveDelays).toHaveBeenCalledTimes(1);
  });

  it('devrait gérer les erreurs de chargement', async () => {
    getLiveDelays.mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Network error');
    expect(console.error).toHaveBeenCalled();
  });

  it('devrait permettre de rafraîchir manuellement', async () => {
    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Premier appel initial
    expect(getLiveDelays).toHaveBeenCalledTimes(1);

    // Rafraîchir manuellement
    result.current.refresh();

    await waitFor(() => {
      expect(getLiveDelays).toHaveBeenCalledTimes(2);
    });
  });

  it('devrait retourner 0 si pas de retards', async () => {
    getLiveDelays.mockResolvedValue({
      delays: [],
      summary: { total: 5, on_time: 5, late: 0, early: 0 },
    });

    const { result } = renderHook(() => useDispatchDelays('2025-10-16'));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.hasDelays).toBe(false);
    expect(result.current.hasCriticalDelays).toBe(false);
    expect(result.current.delayCount).toBe(0);
  });
});
