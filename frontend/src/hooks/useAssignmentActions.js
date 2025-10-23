import { useState, useCallback } from 'react';
import { assignDriver, deleteReservation } from '../services/companyService';
import { retryHttpRequest } from '../utils/retry';

/**
 * Hook personnalisé pour gérer les actions d'assignation avec mise à jour optimiste
 */
export const useAssignmentActions = (onOptimisticUpdate = null, onRollback = null) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleAssignDriver = useCallback(
    async (reservationId, driverId) => {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Mise à jour optimiste si callback fourni
      if (onOptimisticUpdate) {
        onOptimisticUpdate(reservationId, { driver_id: driverId, status: 'assigned' });
      }

      try {
        // Utiliser retry pour plus de robustesse
        await retryHttpRequest(() => assignDriver(reservationId, driverId), {
          retries: 3,
          delay: 1000,
        });
        setSuccess('Chauffeur assigné avec succès');
        return true;
      } catch (err) {
        console.error('[useAssignmentActions] Error assigning driver:', err);
        setError(err.message || "Erreur lors de l'assignation");

        // Rollback en cas d'erreur
        if (onRollback) {
          onRollback(reservationId);
        }

        return false;
      } finally {
        setLoading(false);
      }
    },
    [onOptimisticUpdate, onRollback]
  );

  const handleDeleteReservation = useCallback(
    async (reservationId) => {
      setLoading(true);
      setError(null);
      setSuccess(null);

      // Mise à jour optimiste si callback fourni
      if (onOptimisticUpdate) {
        onOptimisticUpdate(reservationId, { _deleted: true });
      }

      try {
        // Utiliser retry pour plus de robustesse
        await retryHttpRequest(() => deleteReservation(reservationId), {
          retries: 3,
          delay: 1000,
        });
        setSuccess('Réservation supprimée avec succès');
        return true;
      } catch (err) {
        console.error('[useAssignmentActions] Error deleting reservation:', err);
        setError(err.message || 'Erreur lors de la suppression');

        // Rollback en cas d'erreur
        if (onRollback) {
          onRollback(reservationId);
        }

        return false;
      } finally {
        setLoading(false);
      }
    },
    [onOptimisticUpdate, onRollback]
  );

  const clearMessages = useCallback(() => {
    setError(null);
    setSuccess(null);
  }, []);

  return {
    loading,
    error,
    success,
    handleAssignDriver,
    handleDeleteReservation,
    clearMessages,
  };
};
