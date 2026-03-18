// src/hooks/useDriver.js
import { useState, useEffect, useCallback } from 'react';
import {
  fetchCompanyDriversCanonical,
  updateDriverStatus,
  deleteDriver,
} from '../services/companyService';
import { getCompanySocket, joinCompanyRoom } from '../services/companySocket';

const POLL_INTERVAL_MS = 5000;

const mergeDriverLiveUpdate = (driver, update, fromLiveState) => {
  const latitude = update.lat ?? update.latitude ?? update.current_lat ?? driver.latitude;
  const longitude =
    update.lon ?? update.lng ?? update.longitude ?? update.current_lon ?? driver.longitude;
  const locationStatus = fromLiveState
    ? update.location_status ?? driver.location_status ?? null
    : driver.location_status ?? update.location_status ?? null;
  const presenceStatus = fromLiveState
    ? update.presence_status ?? driver.presence_status ?? null
    : driver.presence_status ?? update.presence_status ?? null;
  const status = fromLiveState ? update.status ?? driver.status : driver.status ?? update.status;

  return {
    ...driver,
    latitude,
    longitude,
    location_mode: update.location_mode ?? driver.location_mode ?? null,
    last_seen_seconds: update.last_seen_seconds ?? driver.last_seen_seconds ?? null,
    location_status: locationStatus,
    presence_status: presenceStatus,
    status,
    mission_status: fromLiveState
      ? update.mission_status ?? driver.mission_status ?? null
      : driver.mission_status ?? update.mission_status ?? null,
    recorded_at: update.recorded_at ?? update.timestamp ?? driver.recorded_at ?? null,
    received_at: update.received_at ?? driver.received_at ?? null,
    mission_id: update.mission_id ?? driver.mission_id ?? null,
  };
};

const useDriver = () => {
  const [drivers, setDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const getDrivers = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchCompanyDriversCanonical();
      setDrivers(Array.isArray(data) ? data : []);
      const companyId = Number(data?.[0]?.company_id);
      if (Number.isFinite(companyId) && companyId > 0) {
        joinCompanyRoom(companyId).catch(() => {});
      }
      setError(null);
    } catch (err) {
      console.error(err);
      setError('Erreur lors du chargement des chauffeurs.');
    } finally {
      setLoading(false);
    }
  }, []);

  // CORRECTION : On renomme la fonction pour plus de clarté
  const toggleDriverStatus = useCallback(async (driverId, newStatus) => {
    try {
      await updateDriverStatus(driverId, newStatus);
      // Met à jour l'état local pour un retour visuel immédiat
      setDrivers((prev) =>
        prev.map((d) => (d.id === driverId ? { ...d, is_active: newStatus } : d))
      );
    } catch (err) {
      console.error('Erreur lors de la mise à jour du statut :', err);
    }
  }, []);

  const deleteDriverById = useCallback(async (driverId) => {
    try {
      await deleteDriver(driverId);
      // Met à jour l'état local
      setDrivers((prev) => prev.filter((d) => d.id !== driverId));
    } catch (err) {
      console.error('Erreur lors de la suppression :', err);
    }
  }, []);

  useEffect(() => {
    getDrivers();
  }, [getDrivers]);

  useEffect(() => {
    const socket = getCompanySocket();
    if (!socket) return;

    const applyDelta = (payload, fromLiveState = false) => {
      const driverId = Number(payload?.driver_id ?? payload?.id);
      if (!Number.isFinite(driverId)) return;
      setDrivers((prev) => {
        const index = prev.findIndex((d) => Number(d.id) === driverId);
        if (index < 0) return prev;
        const next = [...prev];
        next[index] = mergeDriverLiveUpdate(next[index], payload, fromLiveState);
        return next;
      });
    };

    const onLiveState = (payload) => applyDelta(payload, true);
    const onLocationUpdate = (payload) => applyDelta(payload, false);
    const onReconnected = () => {
      getDrivers().catch(() => {});
    };

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
    };
  }, [getDrivers]);

  useEffect(() => {
    const poll = () => {
      getDrivers().catch(() => {});
    };
    const intervalId = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [getDrivers]);

  return {
    drivers,
    loading,
    error,
    refreshDrivers: getDrivers,
    toggleDriverStatus,
    deleteDriverById,
  };
};

export default useDriver;
