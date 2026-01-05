// hooks/useOptimisticMutation.js
import { useState, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { detectConflict, resolveConflict, createConflictMessage } from '../utils/conflictResolution';
import { toast } from 'react-toastify';

/**
 * Hook pour mutations optimistes avec rollback et résolution de conflits
 * 
 * @param {Object} options
 * @param {Function} options.mutationFn - Fonction async pour la mutation API
 * @param {Function} options.optimisticUpdate - Fonction pour mettre à jour l'état optimiste
 * @param {Function} options.rollback - Fonction pour annuler la mise à jour optimiste
 * @param {Function} options.onSuccess - Callback appelé en cas de succès
 * @param {Function} options.onError - Callback appelé en cas d'erreur
 * @param {string} options.conflictResolution - Stratégie: 'server-wins', 'client-wins', 'merge', 'user-choice'
 * @param {Function} options.userChoiceCallback - Callback pour 'user-choice' (optionnel)
 * @param {Array<string>} options.queryKeys - Clés React Query à invalider après succès
 * @param {boolean} options.showToast - Afficher les toasts (défaut: true)
 * @param {number} options.retries - Nombre de tentatives en cas d'échec (défaut: 0)
 * @returns {Object} { mutate, isLoading, error, reset }
 */
export function useOptimisticMutation({
  mutationFn,
  optimisticUpdate,
  rollback,
  onSuccess,
  onError,
  conflictResolution = 'server-wins',
  userChoiceCallback = null,
  queryKeys = [],
  showToast = true,
  retries = 0,
}) {
  const queryClient = useQueryClient();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Références pour stocker l'état optimiste et les données originales
  const optimisticStateRef = useRef(null);
  const originalStateRef = useRef(null);
  const mutationDataRef = useRef(null);

  // Fonction de retry avec exponential backoff
  const retryWithBackoff = useCallback(async (fn, attempt = 0) => {
    try {
      return await fn();
    } catch (err) {
      if (attempt < retries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 10000); // Max 10s
        await new Promise(resolve => setTimeout(resolve, delay));
        return retryWithBackoff(fn, attempt + 1);
      }
      throw err;
    }
  }, [retries]);

  // Fonction de mutation principale
  const mutate = useCallback(async (data, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    // Stocker les données de mutation
    mutationDataRef.current = data;
    
    // 1. Mise à jour optimiste immédiate
    if (optimisticUpdate) {
      try {
        // Sauvegarder l'état original pour rollback (si la fonction supporte ce paramètre)
        if (typeof optimisticUpdate === 'function') {
          try {
            originalStateRef.current = optimisticUpdate(data, true); // true = save original
          } catch {
            // Si la fonction ne supporte pas le paramètre saveOriginal, appeler sans
            originalStateRef.current = optimisticUpdate(data);
          }
          
          // Appliquer la mise à jour optimiste
          try {
            optimisticStateRef.current = optimisticUpdate(data, false);
          } catch {
            optimisticStateRef.current = optimisticUpdate(data);
          }
        }
        
        if (showToast && options.showSuccessToast !== false) {
          toast.info('Mise à jour en cours...', { autoClose: 1000 });
        }
      } catch (err) {
        console.error('[useOptimisticMutation] Error in optimistic update:', err);
        // Continuer quand même avec la mutation
      }
    }

    // 2. Appel API en arrière-plan
    try {
      const result = await retryWithBackoff(() => mutationFn(data));
      
      // 3. Vérifier les conflits
      if (optimisticStateRef.current && result) {
        const hasConflict = detectConflict(optimisticStateRef.current, result);
        
        if (hasConflict) {
          console.warn('[useOptimisticMutation] Conflict detected:', {
            optimistic: optimisticStateRef.current,
            server: result,
          });
          
          // Résoudre le conflit selon la stratégie
          const resolvedState = await resolveConflict(
            conflictResolution,
            optimisticStateRef.current,
            result,
            userChoiceCallback
          );
          
          // Mettre à jour avec l'état résolu
          if (optimisticUpdate) {
            optimisticUpdate(resolvedState, false);
          }
          
          // Invalider les queries pour forcer un refetch
          queryKeys.forEach(key => {
            queryClient.invalidateQueries(Array.isArray(key) ? key : [key]);
          });
          
          if (showToast) {
            toast.warning(createConflictMessage(optimisticStateRef.current, result), {
              autoClose: 5000,
            });
          }
        } else {
          // Pas de conflit: utiliser la réponse du serveur
          if (optimisticUpdate) {
            optimisticUpdate(result, false);
          }
        }
      } else if (result) {
        // Pas d'update optimiste, utiliser directement le résultat
        if (optimisticUpdate) {
          optimisticUpdate(result, false);
        }
      }

      // 4. Invalider les queries React Query
      queryKeys.forEach(key => {
        queryClient.invalidateQueries(Array.isArray(key) ? key : [key]);
      });

      // 5. Callback de succès
      if (onSuccess) {
        onSuccess(result, data);
      }
      
      if (showToast && options.showSuccessToast !== false) {
        toast.success(options.successMessage || 'Mise à jour réussie');
      }

      // Nettoyer les refs
      optimisticStateRef.current = null;
      originalStateRef.current = null;
      mutationDataRef.current = null;

      setIsLoading(false);
      return result;
    } catch (err) {
      console.error('[useOptimisticMutation] Mutation failed:', err);
      setError(err);
      
      // 6. Rollback en cas d'erreur
      if (rollback && originalStateRef.current) {
        try {
          rollback(originalStateRef.current, data);
        } catch (rollbackErr) {
          console.error('[useOptimisticMutation] Rollback failed:', rollbackErr);
        }
      } else if (optimisticUpdate && originalStateRef.current) {
        // Fallback: utiliser optimisticUpdate pour restaurer
        try {
          optimisticUpdate(originalStateRef.current, false);
        } catch (restoreErr) {
          console.error('[useOptimisticMutation] Restore failed:', restoreErr);
        }
      }

      // Invalider les queries pour forcer un refetch
      queryKeys.forEach(key => {
        queryClient.invalidateQueries(Array.isArray(key) ? key : [key]);
      });

      // Callback d'erreur
      if (onError) {
        onError(err, data);
      }
      
      if (showToast && options.showErrorToast !== false) {
        const errorMessage = err?.response?.data?.message || err?.message || 'Erreur lors de la mise à jour';
        toast.error(errorMessage);
      }

      // Nettoyer les refs
      optimisticStateRef.current = null;
      originalStateRef.current = null;
      mutationDataRef.current = null;

      setIsLoading(false);
      throw err;
    }
  }, [
    mutationFn,
    optimisticUpdate,
    rollback,
    onSuccess,
    onError,
    conflictResolution,
    userChoiceCallback,
    queryKeys,
    queryClient,
    showToast,
    retryWithBackoff,
  ]);

  // Fonction pour réinitialiser l'état
  const reset = useCallback(() => {
    setError(null);
    setIsLoading(false);
    optimisticStateRef.current = null;
    originalStateRef.current = null;
    mutationDataRef.current = null;
  }, []);

  return {
    mutate,
    isLoading,
    error,
    reset,
  };
}

