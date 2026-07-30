import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getCompanySocket } from '../../services/companySocket';
import { mergeOrUpdateDriverInList, hasExploitableCoords } from '../../utils/mergeDriverLiveUpdate';
import {
  canonicalRealtimeTimeMs,
  shouldAcceptRealtimeEvent,
} from '../../utils/realtimeEventGuard';
import { lirieKeys } from '../../queryKeys/lirie';
import { recordGpsEvent } from '../../utils/companyDashboardPerfInstrumentation';

/** Si le snapshot TanStack est encore frais, pas d’invalidate complet (réduit pics en flapping WS). */
const RECONNECT_MIN_FULL_REFETCH_MS = 60000;
/**
 * Silence sur les événements de position (pas sync/bookings) avant refetch HTTP.
 * Borne 10-15s (Lot 5 perf) : cadence de secours de la carte live quand le socket est down.
 */
const SOCKET_LOCATION_SILENCE_WATCHDOG_MS = 12000;
const WATCHDOG_CHECK_INTERVAL_MS = 5000;
const WATCHDOG_ENABLED = process.env.REACT_APP_COMPANY_REALTIME_WATCHDOG_ENABLED !== 'false';

export function shouldTriggerCompanyDriversWatchdog({
  now,
  lastLocationEventAt,
  lastWatchdogInvalidateAt,
  silenceMs = SOCKET_LOCATION_SILENCE_WATCHDOG_MS,
}) {
  if (now - lastLocationEventAt < silenceMs) return false;
  // Evite les invalidations en boucle pendant un même silence prolongé.
  if (now - lastWatchdogInvalidateAt < silenceMs) return false;
  return true;
}

function trackWatchdogTrigger(companyId) {
  const normalizedCompanyId = companyId == null ? 'unknown' : String(companyId);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('company_realtime_metric', {
        detail: {
          metric: 'company_realtime_watchdog_trigger_total',
          labels: { company_id: normalizedCompanyId },
          value: 1,
          at: Date.now(),
        },
      })
    );
  }
  // eslint-disable-next-line no-console
  console.info(
    JSON.stringify({
      metric: 'company_realtime_watchdog_trigger_total',
      company_id: normalizedCompanyId,
      value: 1,
      timestamp: new Date().toISOString(),
    })
  );
}

/**
 * Fusionne les événements Socket chauffeurs dans le cache TanStack `lirieKeys.companyDrivers()`.
 */
export function useCompanyDriversLiveOverlay(companyId) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;
    let lastLocationSocketEventAt = Date.now();
    let lastWatchdogInvalidateAt = 0;

    const bumpSocketActivity = (payload) => {
      if (hasExploitableCoords(payload)) {
        lastLocationSocketEventAt = Date.now();
      }
    };

    const applyDelta = (payload, fromLiveState) => {
      const accepted = shouldAcceptRealtimeEvent({
        eventId: payload?.event_id,
        entityKey: payload?.driver_id != null ? `driver:${String(payload.driver_id)}` : null,
        canonicalTimeMs: canonicalRealtimeTimeMs(payload),
      });
      if (!accepted) return;

      queryClient.setQueryData(lirieKeys.companyDrivers(), (prev) =>
        mergeOrUpdateDriverInList(prev || [], payload, fromLiveState, companyId ?? null)
      );
    };

    const onLiveState = (payload) => {
      bumpSocketActivity(payload);
      recordGpsEvent();
      applyDelta(payload, true);
    };
    const onLocationUpdate = (payload) => {
      bumpSocketActivity(payload);
      recordGpsEvent();
      applyDelta(payload, false);
    };
    /** Reconnect : invalider seulement si pas de données récentes (overlay socket a déjà mis à jour le cache). */
    const onReconnected = () => {
      const state = queryClient.getQueryState(lirieKeys.companyDrivers());
      const updatedAt = state?.dataUpdatedAt ?? 0;
      if (
        updatedAt > 0 &&
        Date.now() - updatedAt < RECONNECT_MIN_FULL_REFETCH_MS
      ) {
        return;
      }
      queryClient.invalidateQueries({ queryKey: lirieKeys.companyDrivers() });
    };
    const watchdogInterval = WATCHDOG_ENABLED
      ? setInterval(() => {
          // Onglet masqué : pas de refetch GPS en arrière-plan (Lot 5 perf).
          if (typeof document !== 'undefined' && document.hidden) return;
          const now = Date.now();
          if (
            !shouldTriggerCompanyDriversWatchdog({
              now,
              lastLocationEventAt: lastLocationSocketEventAt,
              lastWatchdogInvalidateAt,
              silenceMs: SOCKET_LOCATION_SILENCE_WATCHDOG_MS,
            })
          ) {
            return;
          }
          lastWatchdogInvalidateAt = now;
          trackWatchdogTrigger(companyId);
          queryClient.invalidateQueries({ queryKey: lirieKeys.companyDrivers() });
        }, WATCHDOG_CHECK_INTERVAL_MS)
      : null;

    socket.on('driver_live_state_update', onLiveState);
    socket.on('driver_location_update', onLocationUpdate);
    if (typeof window !== 'undefined') {
      window.addEventListener('company_socket_reconnected', onReconnected);
    }

    return () => {
      socket.off('driver_live_state_update', onLiveState);
      socket.off('driver_location_update', onLocationUpdate);
      if (typeof window !== 'undefined') {
        window.removeEventListener('company_socket_reconnected', onReconnected);
      }
      if (watchdogInterval) clearInterval(watchdogInterval);
    };
  }, [queryClient, companyId]);
}
