import { renderHook, waitFor, act } from '@testing-library/react';
import { useOptimizerStatus } from '../useOptimizerStatus';
import * as dispatchMonitoringService from '../../services/dispatchMonitoringService';

jest.mock('../../services/dispatchMonitoringService');

describe('useOptimizerStatus (P4)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.spyOn(console, 'error').mockImplementation();
  });

  it('single-flight: overlapping loadOptimizerStatus does not parallelize GET', async () => {
    dispatchMonitoringService.getOptimizerStatus.mockResolvedValue({
      running: false,
      company_id: 1,
    });

    const { result } = renderHook(() => useOptimizerStatus({ enabled: true }));

    await waitFor(() => {
      expect(dispatchMonitoringService.getOptimizerStatus).toHaveBeenCalled();
    });
    jest.clearAllMocks();

    let resolveFirst;
    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    dispatchMonitoringService.getOptimizerStatus
      .mockImplementationOnce(() => firstPromise)
      .mockResolvedValue({ running: false, company_id: 1 });

    let p1;
    let p2;
    await act(async () => {
      p1 = result.current.loadOptimizerStatus();
      p2 = result.current.loadOptimizerStatus();
    });

    expect(dispatchMonitoringService.getOptimizerStatus).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveFirst({ running: false, company_id: 1 });
      await p1;
      await p2;
    });

    await waitFor(() => {
      expect(dispatchMonitoringService.getOptimizerStatus).toHaveBeenCalledTimes(2);
    });
  });

  it('exposes optimizer status after successful fetch', async () => {
    dispatchMonitoringService.getOptimizerStatus.mockResolvedValue({
      running: true,
      company_id: 42,
    });

    const { result } = renderHook(() => useOptimizerStatus({ enabled: true }));

    await waitFor(() => {
      expect(result.current.optimizerStatus?.running).toBe(true);
    });
  });
});
