// frontend/src/utils/toast.js
/**
 * Utilitaires pour les notifications toast
 * Utilise sonner pour un feedback utilisateur cohérent
 */

import { toast } from 'sonner';

/**
 * Toast de succès
 */
export const showSuccess = (message, duration = 3000) => {
  return toast.success(message, { duration });
};

/**
 * Toast d'erreur
 */
export const showError = (message, duration = 4000) => {
  return toast.error(message, { duration });
};

/**
 * Toast d'information
 */
export const showInfo = (message, duration = 3000) => {
  return toast.info(message, { duration });
};

/**
 * Toast de warning
 */
export const showWarning = (message, duration = 3500) => {
  return toast.warning(message, { duration });
};

/**
 * Toast de chargement
 */
export const showLoading = (message) => {
  return toast.loading(message);
};

/**
 * Toast avec promise
 * Affiche loading, puis success ou error selon le résultat
 */
export const showPromise = (promise, messages) => {
  return toast.promise(
    promise,
    {
      loading: messages.loading || 'Chargement...',
      success: messages.success || 'Succès !',
      error: messages.error || 'Erreur',
    }
  );
};

/**
 * Fermer tous les toasts
 */
export const dismissAll = () => {
  toast.dismiss();
};

/**
 * Fermer un toast spécifique
 */
export const dismiss = (toastId) => {
  toast.dismiss(toastId);
};

// Export par défaut d'un objet avec toutes les méthodes
const toastUtils = {
  success: showSuccess,
  error: showError,
  info: showInfo,
  warning: showWarning,
  loading: showLoading,
  promise: showPromise,
  dismissAll,
  dismiss,
};

export default toastUtils;
