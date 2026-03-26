import { useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { assignDriver, deleteReservation } from '../services/companyService';
import { LIRIE_QK_PREFIX } from '../queryKeys/lirie';
import { useOptimisticMutation } from './useOptimisticMutation';

/** Préfixe TanStack pour toutes les queries `assigned-reservations` entreprise (voir `lirieKeys.assignedReservations(day)`). */
const ASSIGNED_RESERVATIONS_QUERY_PREFIX = [LIRIE_QK_PREFIX, 'assigned-reservations'];

/** Préfixe pour `lirieKeys.companyReservations(day, scopeHash)` — aligné avec useCompanyData (listScopeHash). */
const COMPANY_RESERVATIONS_QUERY_PREFIX = [LIRIE_QK_PREFIX, 'company-reservations'];

/** Filtre pour setQueriesData / matcher toutes les queries dont la clé commence par `prefix`. */
const prefixFilter = (prefix) => ({ queryKey: prefix, exact: false });

/**
 * Hook personnalisé pour gérer les actions d'assignation avec mise à jour optimiste
 * Utilise useOptimisticMutation pour une gestion complète avec rollback et résolution de conflits
 * 
 * @param {Function|Object} onOptimisticUpdateOrOptions - Callback pour mise à jour optimiste (legacy) OU options object
 * @param {Function} onRollbackOrUndefined - Callback pour rollback (legacy) ou undefined
 * @returns {Object} { loading, error, success, handleAssignDriver, handleDeleteReservation, clearMessages }
 */
export const useAssignmentActions = (onOptimisticUpdateOrOptions = null, onRollbackOrUndefined = null) => {
  // Support backward compatibility: peut être appelé avec (callback1, callback2) ou ({ options })
  let onOptimisticUpdate = null;
  let onRollback = null;
  let conflictResolution = 'server-wins';
  
  if (typeof onOptimisticUpdateOrOptions === 'function') {
    // Legacy pattern: useAssignmentActions(callback1, callback2)
    onOptimisticUpdate = onOptimisticUpdateOrOptions;
    onRollback = onRollbackOrUndefined;
  } else if (onOptimisticUpdateOrOptions && typeof onOptimisticUpdateOrOptions === 'object') {
    // New pattern: useAssignmentActions({ onOptimisticUpdate, onRollback, conflictResolution })
    onOptimisticUpdate = onOptimisticUpdateOrOptions.onOptimisticUpdate || null;
    onRollback = onOptimisticUpdateOrOptions.onRollback || null;
    conflictResolution = onOptimisticUpdateOrOptions.conflictResolution || 'server-wins';
  }
  const queryClient = useQueryClient();

  // Mutation pour assigner un chauffeur
  const assignMutation = useOptimisticMutation({
    mutationFn: async ({ reservationId, driverId }) => {
      return await assignDriver(reservationId, driverId);
    },
    optimisticUpdate: (data, saveOriginal) => {
      const { reservationId, driverId } = data;
      
      // Mise à jour optimiste via React Query (préfixes lirie — voir lirieKeys)
      queryClient.setQueriesData(
        prefixFilter(ASSIGNED_RESERVATIONS_QUERY_PREFIX),
        (oldData) => {
          if (!oldData) return oldData;
          
          if (saveOriginal) {
            // Sauvegarder l'état original pour rollback
            return oldData;
          }
          
          // Mettre à jour optimiste
          return oldData.map((reservation) =>
            reservation.id === reservationId
              ? { ...reservation, driver_id: driverId, driver: { id: driverId }, status: 'assigned' }
              : reservation
          );
        }
      );
      
      queryClient.setQueriesData(
        prefixFilter(COMPANY_RESERVATIONS_QUERY_PREFIX),
        (oldData) => {
          if (!oldData) return oldData;
          
          if (saveOriginal) {
            return oldData;
          }
          
          return oldData.map((reservation) =>
            reservation.id === reservationId
              ? { ...reservation, driver_id: driverId, driver: { id: driverId }, status: 'assigned' }
              : reservation
          );
        }
      );
      
      // Callback legacy pour backward compatibility
      if (onOptimisticUpdate && !saveOriginal) {
        onOptimisticUpdate(reservationId, { driver_id: driverId, status: 'assigned' });
      }
      
      return { reservationId, driverId, status: 'assigned' };
    },
    rollback: (_originalState, variables) => {
      // useOptimisticMutation invalide déjà queryKeys en cas d'erreur ; ici on force la cohérence cache
      queryClient.invalidateQueries(prefixFilter(ASSIGNED_RESERVATIONS_QUERY_PREFIX));
      queryClient.invalidateQueries(prefixFilter(COMPANY_RESERVATIONS_QUERY_PREFIX));

      // Callback legacy pour backward compatibility (variables = payload mutate : { reservationId, driverId })
      if (onRollback) {
        onRollback(variables?.reservationId);
      }
    },
    onSuccess: (_result, _data) => {
      // Invalidation déjà gérée par le hook
    },
    onError: (error, _data) => {
      console.error('[useAssignmentActions] Error assigning driver:', error);
    },
    conflictResolution,
    queryKeys: [ASSIGNED_RESERVATIONS_QUERY_PREFIX, COMPANY_RESERVATIONS_QUERY_PREFIX],
    showToast: true,
    retries: 0,
  });

  // Mutation pour supprimer une réservation
  const deleteMutation = useOptimisticMutation({
    mutationFn: async ({ reservationId, reasonCode, reasonText }) => {
      return await deleteReservation(reservationId, reasonCode, reasonText);
    },
    optimisticUpdate: (data, saveOriginal) => {
      const { reservationId } = data;
      
      // Mise à jour optimiste via React Query
      queryClient.setQueriesData(
        prefixFilter(ASSIGNED_RESERVATIONS_QUERY_PREFIX),
        (oldData) => {
          if (!oldData) return oldData;
          
          if (saveOriginal) {
            return oldData;
          }
          
          // Marquer comme supprimé (ou filtrer)
          return oldData.filter((reservation) => reservation.id !== reservationId);
        }
      );
      
      queryClient.setQueriesData(
        prefixFilter(COMPANY_RESERVATIONS_QUERY_PREFIX),
        (oldData) => {
          if (!oldData) return oldData;
          
          if (saveOriginal) {
            return oldData;
          }
          
          return oldData.filter((reservation) => reservation.id !== reservationId);
        }
      );
      
      // Callback legacy pour backward compatibility
      if (onOptimisticUpdate && !saveOriginal) {
        onOptimisticUpdate(reservationId, { _deleted: true });
      }
      
      return { reservationId, _deleted: true };
    },
    rollback: (originalState, data) => {
      const { reservationId } = data;
      
      queryClient.invalidateQueries(prefixFilter(ASSIGNED_RESERVATIONS_QUERY_PREFIX));
      queryClient.invalidateQueries(prefixFilter(COMPANY_RESERVATIONS_QUERY_PREFIX));
      
      // Callback legacy pour backward compatibility
      if (onRollback) {
        onRollback(reservationId);
      }
    },
    onSuccess: (_result, _data) => {
      // Invalidation déjà gérée par le hook
    },
    onError: (error, _data) => {
      console.error('[useAssignmentActions] Error deleting reservation:', error);
    },
    conflictResolution,
    queryKeys: [ASSIGNED_RESERVATIONS_QUERY_PREFIX, COMPANY_RESERVATIONS_QUERY_PREFIX],
    showToast: true,
    retries: 0,
  });

  const handleAssignDriver = useCallback(
    async (reservationId, driverId) => {
      try {
        await assignMutation.mutate({ reservationId, driverId }, {
          successMessage: 'Chauffeur assigné avec succès',
        });
        return true;
      } catch (err) {
        return false;
      }
    },
    [assignMutation]
  );

  const handleDeleteReservation = useCallback(
    async (reservationId, reasonCode = null, reasonText = null) => {
      await deleteMutation.mutate({ reservationId, reasonCode, reasonText }, {
        successMessage: 'Réservation supprimée avec succès',
      });
    },
    [deleteMutation]
  );

  const clearMessages = useCallback(() => {
    assignMutation.reset();
    deleteMutation.reset();
  }, [assignMutation, deleteMutation]);

  return {
    loading: assignMutation.isLoading || deleteMutation.isLoading,
    error: assignMutation.error || deleteMutation.error,
    success: null, // Géré par les toasts maintenant
    handleAssignDriver,
    handleDeleteReservation,
    clearMessages,
  };
};
