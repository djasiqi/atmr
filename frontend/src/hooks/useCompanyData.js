import { useMemo, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchCompanyReservations, fetchCompanyDriversCanonical } from '../services/companyService';
import { useLirieCompany } from './useLirieCompany';
import { lirieKeys, listScopeHash } from '../queryKeys/lirie';
import { useCompanyDriversLiveOverlay } from './enterprise/useCompanyDriversLiveOverlay';
import { useSocketConnected } from './useCompanySocket';

/** Mode « socket sain » : overlay actif (entreprise connue) + WS connecté — HTTP = snapshot + resync ciblé. */
function useCompanyDriversLiveQueryPolicy(companyId) {
  const socketConnected = useSocketConnected();
  return Boolean(companyId != null && socketConnected);
}

const RESERVATIONS_LIST_SCOPE_HASH = listScopeHash({ flat: true, include_stats: false });

const useCompanyData = ({ day } = {}) => {
  const queryClient = useQueryClient();
  const { company, loadingCompany, companyError, reloadCompany } = useLirieCompany();

  const reservationDayKey = day ?? '__all__';

  const driversLiveHealthy = useCompanyDriversLiveQueryPolicy(company?.id);

  const companyDriversQueryOptions = useMemo(() => {
    if (driversLiveHealthy) {
      return {
        staleTime: Infinity,
        refetchOnWindowFocus: false,
        /** Resync reconnect : `useCompanyDriversLiveOverlay` + `company_socket_reconnected` → invalidateQueries (éviter doublon avec refetch online). */
        refetchOnReconnect: false,
      };
    }
    return {
      staleTime: 45_000,
      refetchOnWindowFocus: true,
      refetchOnReconnect: true,
    };
  }, [driversLiveHealthy]);

  const {
    data: reservations = [],
    isLoading: loadingReservations,
    refetch: refetchReservations,
    error: reservationsQueryError,
  } = useQuery({
    queryKey: lirieKeys.companyReservations(reservationDayKey, RESERVATIONS_LIST_SCOPE_HASH),
    queryFn: async () => {
      const data = await fetchCompanyReservations(day, { fields: 'dashboard' });
      return Array.isArray(data) ? data : (data?.reservations ?? []);
    },
    staleTime: 30_000,
  });

  const {
    data: driver = [],
    isLoading: loadingDriver,
    refetch: refetchDriver,
    error: driversQueryError,
  } = useQuery({
    queryKey: lirieKeys.companyDrivers(),
    queryFn: async () => {
      const data = await fetchCompanyDriversCanonical();
      return Array.isArray(data) ? data : data?.driver ?? [];
    },
    ...companyDriversQueryOptions,
  });

  useCompanyDriversLiveOverlay(company?.id);

  const combinedError = useMemo(() => {
    if (companyError) return companyError;
    const re = reservationsQueryError;
    const de = driversQueryError;
    if (re?.code === 'ECONNABORTED' || re?.message?.includes?.('timeout')) {
      return 'La récupération des réservations a pris trop de temps. Veuillez réessayer.';
    }
    if (de?.code === 'ECONNABORTED' || de?.message?.includes?.('timeout')) {
      return 'La récupération des chauffeurs a pris trop de temps. Veuillez réessayer.';
    }
    if (re && re?.response?.status !== 403 && re?.response?.status !== 404 && re?.response?.status !== 401) {
      return 'Erreur lors du chargement des réservations.';
    }
    if (de && de?.response?.status !== 403 && de?.response?.status !== 404 && de?.response?.status !== 401) {
      return 'Erreur lors du chargement des chauffeurs.';
    }
    return null;
  }, [companyError, reservationsQueryError, driversQueryError]);

  const reloadReservations = useCallback(async () => {
    await refetchReservations();
  }, [refetchReservations]);

  const reloadDriver = useCallback(() => refetchDriver(), [refetchDriver]);

  const upsertReservation = useCallback(
    (booking) => {
      if (!booking || booking.id == null) return;
      const key = lirieKeys.companyReservations(reservationDayKey, RESERVATIONS_LIST_SCOPE_HASH);
      queryClient.setQueryData(key, (prev) => {
        const list = Array.isArray(prev) ? prev : [];
        const id = Number(booking.id);
        const idx = list.findIndex((r) => Number(r.id) === id);
        if (idx >= 0) {
          const next = [...list];
          next[idx] = { ...next[idx], ...booking };
          return next;
        }
        return [booking, ...list];
      });
    },
    [queryClient, reservationDayKey]
  );

  return {
    company,
    reservations,
    driver,
    loadingCompany,
    loadingReservations,
    loadingDriver,
    error: combinedError,
    reloadCompany,
    reloadReservations,
    reloadDriver,
    upsertReservation,
  };
};

export default useCompanyData;
