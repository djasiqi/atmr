// hooks/useHybridDataSync.js
import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Hook pour synchronisation hybride: Socket.IO (primary) + Polling (fallback/safety net)
 * 
 * Stratégie:
 * - Socket connecté + données récentes (< 2 min): Pas de polling
 * - Socket connecté + données stale (> 2 min): Poll toutes les 2-5 minutes
 * - Socket déconnecté: Poll toutes les 30-60 secondes
 * - Échec de poll: Exponential backoff, max 5 minutes
 * 
 * @param {Object} options
 * @param {Function} options.fetchFn - Fonction async pour récupérer les données
 * @param {Object} options.socket - Instance Socket.IO (optionnel)
 * @param {number} options.staleThreshold - Seuil de staleness en ms (défaut: 120000 = 2 min)
 * @param {number} options.pollIntervalConnected - Intervalle de polling quand socket connecté (défaut: 180000 = 3 min)
 * @param {number} options.pollIntervalDisconnected - Intervalle de polling quand socket déconnecté (défaut: 45000 = 45s)
 * @param {Function} options.onUpdate - Callback appelé quand les données sont mises à jour
 * @param {Array} options.dependencies - Dépendances pour le hook (ex: [date])
 * @returns {Object} { lastUpdate, isPolling, pollError }
 */
export function useHybridDataSync({
  fetchFn,
  socket = null,
  staleThreshold = 120000, // 2 minutes
  pollIntervalConnected = 180000, // 3 minutes
  pollIntervalDisconnected = 45000, // 45 seconds
  onUpdate = null,
  dependencies = [],
}) {
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [isPolling, setIsPolling] = useState(false);
  const [pollError, setPollError] = useState(null);
  
  const lastUpdateRef = useRef(Date.now());
  const pollTimeoutRef = useRef(null);
  const backoffDelayRef = useRef(1000); // Start with 1 second
  const consecutiveFailuresRef = useRef(0);
  const maxBackoffDelay = 300000; // 5 minutes max

  // Mettre à jour lastUpdate et notifier
  const updateLastUpdate = useCallback((timestamp = Date.now()) => {
    lastUpdateRef.current = timestamp;
    setLastUpdate(timestamp);
    if (onUpdate) {
      onUpdate(timestamp);
    }
  }, [onUpdate]);

  // Fonction de polling avec exponential backoff
  const performPoll = useCallback(async () => {
    if (!fetchFn) return;

    setIsPolling(true);
    setPollError(null);

    try {
      const result = await fetchFn();
      // Succès: réinitialiser le backoff
      backoffDelayRef.current = 1000;
      consecutiveFailuresRef.current = 0;
      updateLastUpdate();
      
      console.log(JSON.stringify({
        event: 'hybrid_poll_success',
        timestamp: new Date().toISOString(),
        socket_connected: socket?.connected || false,
      }));
      
      return result;
    } catch (error) {
      // Échec: augmenter le backoff
      consecutiveFailuresRef.current += 1;
      backoffDelayRef.current = Math.min(
        backoffDelayRef.current * 1.5,
        maxBackoffDelay
      );
      
      setPollError(error);
      
      console.warn(JSON.stringify({
        event: 'hybrid_poll_error',
        error: error?.message || String(error),
        consecutive_failures: consecutiveFailuresRef.current,
        next_backoff_ms: backoffDelayRef.current,
        timestamp: new Date().toISOString(),
      }));
      
      throw error;
    } finally {
      setIsPolling(false);
    }
  }, [fetchFn, socket, updateLastUpdate]);

  // Déterminer l'intervalle de polling selon l'état
  const getPollInterval = useCallback(() => {
    const isSocketConnected = socket?.connected || false;
    const timeSinceLastUpdate = Date.now() - lastUpdateRef.current;
    const isDataStale = timeSinceLastUpdate > staleThreshold;

    if (!isSocketConnected) {
      // Socket déconnecté: poll fréquemment
      return pollIntervalDisconnected;
    } else if (isDataStale) {
      // Socket connecté mais données stale: poll modérément
      return pollIntervalConnected;
    } else {
      // Socket connecté et données fraîches: pas de polling nécessaire
      return null;
    }
  }, [socket, staleThreshold, pollIntervalConnected, pollIntervalDisconnected]);

  // Planifier le prochain poll
  const scheduleNextPoll = useCallback(() => {
    // Nettoyer le timeout précédent
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }

    const interval = getPollInterval();
    
    if (interval === null) {
      // Pas de polling nécessaire pour l'instant
      return;
    }

    // Utiliser le backoff delay si on a eu des échecs récents
    const delay = consecutiveFailuresRef.current > 0 
      ? Math.min(interval, backoffDelayRef.current)
      : interval;

    pollTimeoutRef.current = setTimeout(() => {
      performPoll()
        .then(() => {
          // Réessayer après l'intervalle normal
          scheduleNextPoll();
        })
        .catch(() => {
          // En cas d'erreur, réessayer avec backoff
          scheduleNextPoll();
        });
    }, delay);
  }, [getPollInterval, performPoll]);

  // Effet principal: démarrer le polling hybride
  useEffect(() => {
    // Poll initial si nécessaire
    const timeSinceLastUpdate = Date.now() - lastUpdateRef.current;
    const isDataStale = timeSinceLastUpdate > staleThreshold;
    const isSocketConnected = socket?.connected || false;

    if (!isSocketConnected || isDataStale) {
      // Poll immédiat si socket déconnecté ou données stale
      performPoll().then(() => {
        scheduleNextPoll();
      }).catch(() => {
        scheduleNextPoll();
      });
    } else {
      // Sinon, planifier le prochain poll
      scheduleNextPoll();
    }

    // Écouter les changements de connexion socket
    const onConnect = () => {
      console.log(JSON.stringify({
        event: 'hybrid_socket_connected',
        timestamp: new Date().toISOString(),
      }));
      // Réinitialiser le backoff quand socket se reconnecte
      backoffDelayRef.current = 1000;
      consecutiveFailuresRef.current = 0;
      scheduleNextPoll();
    };

    const onDisconnect = () => {
      console.log(JSON.stringify({
        event: 'hybrid_socket_disconnected',
        timestamp: new Date().toISOString(),
      }));
      // Poll immédiatement quand socket se déconnecte
      performPoll().then(() => {
        scheduleNextPoll();
      }).catch(() => {
        scheduleNextPoll();
      });
    };

    if (socket) {
      socket.on('connect', onConnect);
      socket.on('disconnect', onDisconnect);
    }

    // Cleanup
    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
        pollTimeoutRef.current = null;
      }
      if (socket) {
        socket.off('connect', onConnect);
        socket.off('disconnect', onDisconnect);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [socket, staleThreshold, performPoll, scheduleNextPoll, ...dependencies]);

  return {
    lastUpdate,
    isPolling,
    pollError,
    // Fonction manuelle pour forcer un poll
    forcePoll: performPoll,
  };
}

