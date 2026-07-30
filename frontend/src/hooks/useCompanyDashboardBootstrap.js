// frontend/src/hooks/useCompanyDashboardBootstrap.js
import { useEffect, useMemo, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchCompanyDashboardBootstrap } from '../services/companyService';
import { lirieKeys, listScopeHash } from '../queryKeys/lirie';
import { getCurrentAuthEnv } from '../utils/apiClient';
import { setSnapshotCursor } from '../utils/companyRealtimeSequenceGate';

/** Doit rester identique au scope utilisé par useCompanyData / CompanyDashboard. */
const RESERVATIONS_DASHBOARD_SCOPE_HASH = listScopeHash({ flat: true, include_stats: false });

/**
 * Bootstrap unique du dashboard entreprise (Lot 3 perf company-space) :
 * charge en **1 GET** le KPI du jour, les réservations (projection dashboard),
 * le mode dispatch, le résumé notifications et le `snapshot_cursor` temps réel —
 * voir docs/perf-company-space-lot3-dashboard.md et
 * `backend/routes/companies.py::CompanyDashboardBootstrap`.
 *
 * Alimente ensuite (`setQueryData`) les caches TanStack déjà consommés par
 * `useCompanyData` (réservations du jour), `useDispatchMode` et
 * `CompanyNotificationBell` (badge non-lues) pour éviter les GET redondants.
 *
 * @param {string|null} day - YYYY-MM-DD
 * @param {{ companyId?: number|string|null, enabled?: boolean }} [options]
 */
export function useCompanyDashboardBootstrap(day, { companyId = null, enabled = true } = {}) {
  const queryClient = useQueryClient();
  const authEnv = getCurrentAuthEnv();
  // Évite de ré-appliquer le même snapshot plusieurs fois (re-render sans nouvelle donnée).
  const appliedStampRef = useRef(null);

  const queryKey = useMemo(
    () => lirieKeys.companyDashboardBootstrap(authEnv, companyId, day),
    [authEnv, companyId, day]
  );

  const {
    data,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey,
    queryFn: () => fetchCompanyDashboardBootstrap(day),
    enabled: Boolean(enabled && companyId && day),
    staleTime: 15_000,
    retry: 1,
  });

  useEffect(() => {
    if (!data || companyId == null) return;
    const stamp = `${companyId}:${day}:${data.generated_at || ''}:${data.snapshot_cursor ?? ''}`;
    if (appliedStampRef.current === stamp) return;
    appliedStampRef.current = stamp;

    // Curseur temps réel — null = Redis dégradé (jamais 0 faux-sain).
    const rtHealth = data.health?.realtime_sequence;
    setSnapshotCursor(companyId, data.snapshot_cursor, {
      degraded: rtHealth === 'degraded' || data.snapshot_cursor == null,
    });
    if (rtHealth !== 'degraded' && data.snapshot_cursor != null) {
      const { clearResyncAfterBootstrapSuccess } = require('../utils/companyRealtimeSequenceGate');
      clearResyncAfterBootstrapSuccess(companyId);
    }

    // Réservations du jour : même clé/scope que useCompanyData (day ?? '__all__') → pas de second GET.
    if (Array.isArray(data.bookings)) {
      queryClient.setQueryData(
        lirieKeys.companyReservations(day ?? '__all__', RESERVATIONS_DASHBOARD_SCOPE_HASH),
        data.bookings
      );
    }

    // Mode dispatch (useDispatchMode) — évite GET /company_dispatch/mode séparé.
    if (data.dispatch_mode) {
      queryClient.setQueryData(lirieKeys.dispatchMode(), data.dispatch_mode);
    }

    // Résumé KPI jour, aligné sur GET /me/reservations/summary (consommateurs légers).
    if (data.kpi) {
      queryClient.setQueryData(lirieKeys.companyReservationsSummary(day), {
        date: data.date || day,
        stats: data.kpi,
        generated_at: data.generated_at,
      });
    }

    // Badge notifications (cloche header) : pré-alimenté sans GET séparé au montage.
    if (data.notifications && typeof data.notifications.unread_count === 'number') {
      const unreadBadgeKey = [...lirieKeys.companyNotifications(companyId), 'unread-badge'];
      queryClient.setQueryData(unreadBadgeKey, (prev) => ({
        notifications: prev?.notifications || [],
        unread_count: data.notifications.unread_count,
        total: prev?.total || 0,
      }));
    }
  }, [data, companyId, day, queryClient]);

  return {
    bootstrap: data ?? null,
    isBootstrapLoading: isLoading,
    isBootstrapFetching: isFetching,
    isBootstrapError: isError,
    bootstrapError: error,
    refetchBootstrap: refetch,
    snapshotCursor: data?.snapshot_cursor ?? null,
  };
}

export default useCompanyDashboardBootstrap;
