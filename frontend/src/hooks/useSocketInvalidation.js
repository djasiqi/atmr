import { useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

/**
 * Hook réutilisable pour invalider automatiquement React Query sur événements Socket.IO.
 *
 * Simplifie l'invalidation des queries React Query lors de la réception d'événements Socket.IO.
 * Gère automatiquement l'écoute des événements et le cleanup.
 *
 * @param {Object} socket - Instance Socket.IO (peut être null/undefined)
 * @param {Object} eventMap - Mapping des événements Socket.IO vers les query keys à invalider
 * @param {Object} options - Options supplémentaires
 * @param {Function} options.onEvent - Callback optionnel appelé à chaque événement reçu
 * @param {Array} options.dependencies - Dépendances supplémentaires pour le useEffect
 *
 * @example
 * ```javascript
 * useSocketInvalidation(socket, {
 *   'booking:assigned': [['reservations'], ['assigned-reservations', date]],
 *   'booking:updated': [['reservations']],
 *   'dispatch:assignment:created': (data) => [
 *     ['assigned-reservations', data.date || dispatchDay],
 *     ['reservations'],
 *   ],
 * });
 * ```
 *
 * @example Avec callback
 * ```javascript
 * useSocketInvalidation(socket, {
 *   'booking:assigned': [['reservations']],
 * }, {
 *   onEvent: (eventName, data) => {
 *     console.log(`Event ${eventName} received:`, data);
 *   },
 * });
 * ```
 */
export function useSocketInvalidation(socket, eventMap, options = {}) {
  const queryClient = useQueryClient();
  const { onEvent, dependencies = [] } = options;

  // ✅ useRef pour isoler seenEventIds par instance du hook (évite partage global)
  const seenEventIdsRef = useRef(new Set());
  const MAX_SEEN_EVENTS = 1000;

  // ✅ Queue d'invalidation pour coalescing (évite les storms)
  const invalidationQueueRef = useRef(new Set());
  const invalidationTimeoutRef = useRef(null);

  useEffect(() => {
    if (!socket || !eventMap || typeof eventMap !== 'object') {
      return;
    }

    // Créer les handlers pour chaque événement
    const handlers = {};
    const seenEventIds = seenEventIdsRef.current;
    const invalidationQueue = invalidationQueueRef.current;

    // ✅ Fonction pour flush les invalidations en batch
    const flushInvalidations = () => {
      const keysToInvalidate = Array.from(invalidationQueue);
      invalidationQueue.clear();
      
      // Désérialiser les query keys et invalider
      keysToInvalidate.forEach((serializedKey) => {
        try {
          const queryKey = JSON.parse(serializedKey);
          queryClient.invalidateQueries({ queryKey });
        } catch (error) {
          console.warn(
            `[useSocketInvalidation] Failed to parse query key: ${serializedKey}`,
            error
          );
        }
      });
      
      invalidationTimeoutRef.current = null;
    };

    Object.keys(eventMap).forEach((eventName) => {
      const queryKeysConfig = eventMap[eventName];

      handlers[eventName] = (data) => {
        // ✅ Déduplication par event_id (AVANT onEvent et invalidation)
        const eventId = data?.event_id;
        if (eventId) {
          if (seenEventIds.has(eventId)) {
            console.warn(
              `[useSocketInvalidation] Duplicate event ${eventId} ignored for event "${eventName}"`
            );
            return;
          }
          seenEventIds.add(eventId);
          // Limiter la taille du Set (garder les 1000 derniers - FIFO)
          if (seenEventIds.size > MAX_SEEN_EVENTS) {
            const first = seenEventIds.values().next().value;
            seenEventIds.delete(first);
          }
        }

        // Appeler le callback optionnel
        if (onEvent) {
          onEvent(eventName, data);
        }

        // Déterminer les query keys à invalider
        let queryKeysToInvalidate = [];

        if (typeof queryKeysConfig === 'function') {
          // Si c'est une fonction, l'appeler avec les données de l'événement
          queryKeysToInvalidate = queryKeysConfig(data);
        } else if (Array.isArray(queryKeysConfig)) {
          // Si c'est un tableau, l'utiliser directement
          queryKeysToInvalidate = queryKeysConfig;
        } else {
          console.warn(
            `[useSocketInvalidation] Invalid config for event "${eventName}": expected array or function, got ${typeof queryKeysConfig}`
          );
          return;
        }

        // S'assurer que queryKeysToInvalidate est un tableau de tableaux
        if (!Array.isArray(queryKeysToInvalidate)) {
          console.warn(
            `[useSocketInvalidation] Invalid return value for event "${eventName}": expected array of arrays, got ${typeof queryKeysToInvalidate}`
          );
          return;
        }

        // ✅ Ajouter les query keys à la queue d'invalidation (coalescing)
        queryKeysToInvalidate.forEach((queryKey) => {
          if (!Array.isArray(queryKey)) {
            console.warn(
              `[useSocketInvalidation] Invalid query key for event "${eventName}": expected array, got ${typeof queryKey}`
            );
            return;
          }

          // Sérialiser la query key pour l'ajouter au Set (évite les doublons)
          const serializedKey = JSON.stringify(queryKey);
          invalidationQueue.add(serializedKey);
        });

        // ✅ Coalescer: flush après 100ms de silence (évite les storms)
        if (invalidationTimeoutRef.current) {
          clearTimeout(invalidationTimeoutRef.current);
        }
        invalidationTimeoutRef.current = setTimeout(flushInvalidations, 100);
      };

      // Écouter l'événement
      socket.on(eventName, handlers[eventName]);
    });

    // Cleanup : retirer tous les listeners et nettoyer le timeout
    return () => {
      Object.keys(handlers).forEach((eventName) => {
        socket.off(eventName, handlers[eventName]);
      });
      
      // ✅ Nettoyer le timeout et flush les invalidations restantes
      if (invalidationTimeoutRef.current) {
        clearTimeout(invalidationTimeoutRef.current);
        invalidationTimeoutRef.current = null;
      }
      // Flush les invalidations restantes avant le démontage
      if (invalidationQueue.size > 0) {
        flushInvalidations();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [socket, queryClient, onEvent, ...dependencies]);
}

export default useSocketInvalidation;

