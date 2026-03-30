import { useState, useCallback, useEffect, useRef } from 'react';
import { getLiveDelays } from '../services/dispatchMonitoringService';

/** Debounce avant GET /delays/live après invalidation socket (P3). */
const SCHEDULE_DEBOUNCE_MS = 400;

/** Fenêtre min entre deux GET pour une même date (hors loadDelays forcé) — coalesce les rafales. */
const MIN_GET_INTERVAL_MS = 15000;

/**
 * Hook pour les retards en temps réel (GET /company_dispatch/delays/live).
 * Réservé COMPANY/ADMIN. Passer enabled: false si rôle DRIVER pour éviter 403.
 *
 * P3 : écoute `delay_live_invalidate` (room company), debounce + single-flight ;
 * plus de polling 30s côté page si socket connecté (voir UnifiedDispatch).
 *
 * @param {string} date - Date au format YYYY-MM-DD
 * @param {boolean} enabled - Si false, aucun fetch. Défaut true.
 * @param {{ socket?: import('socket.io-client').Socket | null }} [options]
 */
export const useLiveDelays = (date, enabled = true, options = {}) => {
  const { socket = null } = options;
  const [delays, setDelays] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const inflightRef = useRef(false);
  const pendingRef = useRef(false);
  const debounceTimerRef = useRef(null);
  /** Dernier GET réussi (ms) — coalescing invalidations socket. */
  const lastSuccessfulFetchAtRef = useRef(0);

  const runLoadDelays = useCallback(async (opts = {}) => {
    const force = opts.force === true;
    /** Fenêtre min uniquement pour invalidations socket (schedule), pas pour refresh explicite. */
    const coalesce = opts.coalesce === true;
    if (!force && coalesce) {
      const now = Date.now();
      const last = lastSuccessfulFetchAtRef.current;
      if (last > 0 && now - last < MIN_GET_INTERVAL_MS) {
        return;
      }
    }
    if (inflightRef.current) {
      pendingRef.current = true;
      return;
    }
    inflightRef.current = true;
    setLoading(true);
    setError(null);

    try {
      const response = await getLiveDelays(date);
      if (response) {
        setDelays(response.delays || []);
        setSummary(response.summary || null);
        lastSuccessfulFetchAtRef.current = Date.now();
      }
    } catch (err) {
      if (err?.response?.status === 401 && err?.config?._retryAfterRefresh) {
        return;
      }
      if (err?.response?.status !== 401) {
        console.error('[useLiveDelays] Error loading delays:', err);
        setError(err.message || 'Erreur lors du chargement des retards');
      } else {
        console.debug('[useLiveDelays] 401 error, refresh token will be attempted');
      }
    } finally {
      setLoading(false);
      inflightRef.current = false;
      if (pendingRef.current) {
        pendingRef.current = false;
        await runLoadDelays({ force: true });
      }
    }
  }, [date]);

  /** GET debouncé (invalidations socket, rafales). */
  const scheduleLoadDelays = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null;
      runLoadDelays({ coalesce: true });
    }, SCHEDULE_DEBOUNCE_MS);
  }, [runLoadDelays]);

  /**
   * GET immédiat (montage, refresh utilisateur, fin de run) — single-flight interne.
   */
  const loadDelays = useCallback(async () => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }
    await runLoadDelays({ force: true });
  }, [runLoadDelays]);

  useEffect(() => {
    lastSuccessfulFetchAtRef.current = 0;
  }, [date]);

  useEffect(() => {
    if (!socket || !enabled || !date) {
      return undefined;
    }
    const onInvalidate = (payload) => {
      const d = payload?.date;
      if (d && d !== date) {
        return;
      }
      scheduleLoadDelays();
    };
    socket.on('delay_live_invalidate', onInvalidate);
    return () => {
      socket.off('delay_live_invalidate', onInvalidate);
    };
  }, [socket, date, enabled, scheduleLoadDelays]);

  useEffect(() => {
    if (enabled && date) {
      loadDelays();
    }
  }, [enabled, date, loadDelays]);

  return {
    delays,
    summary,
    loading,
    error,
    loadDelays,
    scheduleLoadDelays,
    setDelays,
    setSummary,
  };
};
