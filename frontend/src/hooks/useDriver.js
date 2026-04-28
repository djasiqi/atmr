// src/hooks/useDriver.js
import { useCallback, useEffect, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchCompanyDriversCanonical,
  updateDriverStatus,
  deleteDriver,
} from '../services/companyService';
import { joinCompanyRoom } from '../services/companySocket';
import { useLirieCompany } from './useLirieCompany';
import { useSocketConnected } from './useCompanySocket';
import { useCompanyDriversLiveOverlay } from './enterprise/useCompanyDriversLiveOverlay';
import { lirieKeys } from '../queryKeys/lirie';

const POLL_INTERVAL_MS = 5000;

const useDriver = () => {
  const queryClient = useQueryClient();
  const { company } = useLirieCompany();
  const socketConnected = useSocketConnected();
  const driversLiveHealthy = Boolean(company?.id != null && socketConnected);

  const companyDriversQueryOptions = useMemo(
    () =>
      driversLiveHealthy
        ? {
            staleTime: Infinity,
            refetchOnWindowFocus: false,
            refetchOnReconnect: false,
          }
        : {
            staleTime: 45_000,
            refetchOnWindowFocus: true,
            refetchOnReconnect: true,
          },
    [driversLiveHealthy]
  );

  const {
    data: drivers = [],
    isLoading,
    isRefetching,
    error: queryError,
    refetch,
  } = useQuery({
    queryKey: lirieKeys.companyDrivers(),
    queryFn: async () => {
      const data = await fetchCompanyDriversCanonical();
      return Array.isArray(data) ? data : data?.driver ?? [];
    },
    ...companyDriversQueryOptions,
  });

  useCompanyDriversLiveOverlay(company?.id);

  // Room Socket `company_{id}` : dès l’id entreprise connu (ne pas attendre la liste : liste vide = pas de drivers[0])
  useEffect(() => {
    const fromContext = company?.id != null ? Number(company.id) : null;
    const fromDriver = Number(drivers[0]?.company_id);
    const companyId = Number.isFinite(fromContext) && fromContext > 0 ? fromContext : fromDriver;
    if (Number.isFinite(companyId) && companyId > 0) {
      joinCompanyRoom(companyId).catch(() => {});
    }
  }, [company?.id, drivers]);

  useEffect(() => {
    const poll = () => {
      if (driversLiveHealthy) return;
      queryClient.invalidateQueries({ queryKey: lirieKeys.companyDrivers() });
    };
    const intervalId = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [driversLiveHealthy, queryClient]);

  const refreshDrivers = useCallback(() => refetch(), [refetch]);

  const toggleDriverStatus = useCallback(
    async (driverId, newStatus) => {
      try {
        await updateDriverStatus(driverId, newStatus);
        queryClient.setQueryData(lirieKeys.companyDrivers(), (prev) => {
          const list = Array.isArray(prev) ? prev : [];
          const id = Number(driverId);
          return list.map((d) => (Number(d.id) === id ? { ...d, is_active: newStatus } : d));
        });
      } catch (err) {
        console.error('Erreur lors de la mise à jour du statut :', err);
      }
    },
    [queryClient]
  );

  const deleteDriverById = useCallback(
    async (driverId) => {
      try {
        await deleteDriver(driverId);
        queryClient.setQueryData(lirieKeys.companyDrivers(), (prev) => {
          const list = Array.isArray(prev) ? prev : [];
          const id = Number(driverId);
          return list.filter((d) => Number(d.id) !== id);
        });
      } catch (err) {
        console.error('Erreur lors de la suppression :', err);
      }
    },
    [queryClient]
  );

  const errorMessage = queryError ? 'Erreur lors du chargement des chauffeurs.' : null;

  return {
    drivers,
    loading: isLoading,
    isRefetching,
    error: errorMessage,
    refreshDrivers,
    toggleDriverStatus,
    deleteDriverById,
  };
};

export default useDriver;
