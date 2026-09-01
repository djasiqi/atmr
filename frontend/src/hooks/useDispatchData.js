import { useState, useCallback } from 'react';
import { fetchAssignedReservations, fetchCompanyReservations } from '../services/companyService';
import { isReturnLeg } from '../utils/bookingScheduling';

/**
 * Hook personnalisé pour gérer le chargement des données de dispatch
 */
export const useDispatchData = (date, dispatchMode) => {
  const [dispatches, setDispatches] = useState([]);
  // true: évite un frame « vide / aucune course » avant le premier fetch
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadDispatches = useCallback(async () => {
    if (dispatchMode == null) {
      return;
    }
    setLoading(true);
    setError(null);

    try {
      // En mode Manuel : charger TOUTES les réservations du jour (pas seulement les assignées)
      // En modes Semi-Auto et Fully-Auto : charger les réservations assignées/dispatched
      const data =
        dispatchMode === 'manual'
          ? await fetchCompanyReservations(date)
          : await fetchAssignedReservations(date);

      const dispatches = Array.isArray(data) ? data : data?.data || [];
      setDispatches(dispatches);

      // Debug: Vérifier les données reçues (en mode développement uniquement)
      if (process.env.NODE_ENV === 'development') {
        console.log('[useDispatchData] Dispatches loaded:', dispatches.length);

        // Vérifier les retours avec heure à confirmer
        const returnsToConfirm = dispatches.filter(
          (d) => isReturnLeg(d) && (d.time_confirmed === false || !d.scheduled_time)
        );

        if (returnsToConfirm.length > 0) {
          console.log('[useDispatchData] Retours avec heure à confirmer:', returnsToConfirm.length);
          returnsToConfirm.forEach((r) => {
            console.log(
              `  - ID ${r.id}: time_confirmed=${r.time_confirmed}, scheduled_time=${r.scheduled_time}`
            );
          });
        }
      }
    } catch (err) {
      console.error('[useDispatchData] Error loading dispatches:', err);
      setError(err.message || 'Erreur lors du chargement des données');
    } finally {
      setLoading(false);
    }
  }, [date, dispatchMode]);

  return {
    dispatches,
    loading,
    error,
    loadDispatches,
    setDispatches,
  };
};
