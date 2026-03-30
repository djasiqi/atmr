import { useState, useCallback, useEffect, useRef } from 'react';
import { getOptimizerStatus } from '../services/dispatchMonitoringService';

/**
 * P4 — polling adaptatif pour GET /company_dispatch/optimizer/status.
 * check_interval_seconds (API) is intentionally not used for client poll cadence in v1 — fixed intervals.
 */
const P4_OPTIMIZER_POLL_RUNNING_MS = 30000;
const P4_OPTIMIZER_POLL_STABLE_MS = 180000;

/**
 * @param {{ enabled?: boolean }} [options]
 * @returns {{
 *   optimizerStatus: object | null,
 *   loadOptimizerStatus: () => Promise<object | null | undefined>,
 * }}
 */
export function useOptimizerStatus({ enabled = true } = {}) {
  const [optimizerStatus, setOptimizerStatus] = useState(null);
  const inflightRef = useRef(false);
  const pendingRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runLoadOptimizerStatus = useCallback(async () => {
    const isDevelopment = process.env.NODE_ENV === 'development';
    if (isDevelopment) {
      if (mountedRef.current) {
        setOptimizerStatus(null);
      }
      return null;
    }

    if (inflightRef.current) {
      pendingRef.current = true;
      return undefined;
    }
    inflightRef.current = true;
    try {
      const status = await getOptimizerStatus();
      if (mountedRef.current && status) {
        setOptimizerStatus(status);
      }
      return status ?? null;
    } catch (err) {
      if (err?.response?.status === 401 && err?.config?._retryAfterRefresh) {
        return null;
      }
      if (err?.response?.status !== 401) {
        console.error('[useOptimizerStatus] Error loading optimizer:', err);
      } else {
        console.debug('[useOptimizerStatus] 401 error, refresh token will be attempted');
      }
      return null;
    } finally {
      inflightRef.current = false;
      if (pendingRef.current && mountedRef.current) {
        pendingRef.current = false;
        await runLoadOptimizerStatus();
      }
    }
  }, []);

  const loadOptimizerStatus = useCallback(async () => {
    return runLoadOptimizerStatus();
  }, [runLoadOptimizerStatus]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    loadOptimizerStatus();
  }, [enabled, loadOptimizerStatus]);

  const pollMs =
    optimizerStatus?.running === true ? P4_OPTIMIZER_POLL_RUNNING_MS : P4_OPTIMIZER_POLL_STABLE_MS;

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    if (process.env.NODE_ENV === 'development') {
      return undefined;
    }
    const id = setInterval(() => {
      loadOptimizerStatus();
    }, pollMs);
    return () => clearInterval(id);
  }, [enabled, optimizerStatus?.running, pollMs, loadOptimizerStatus]);

  return {
    optimizerStatus,
    loadOptimizerStatus,
  };
}
