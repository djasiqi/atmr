// frontend/src/utils/toast.js
/**
 * Utilitaires pour les notifications toast
 * Utilise react-hot-toast pour un feedback utilisateur moderne
 */

import toast from 'react-hot-toast';

/**
 * Toast de succès
 */
export const showSuccess = (message, duration = 3000) => {
  return toast.success(message, {
    duration,
    position: 'top-right',
    style: {
      background: '#d4edda',
      color: '#155724',
      border: '1px solid #c3e6cb',
      borderLeft: '4px solid #28a745',
      padding: '16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
    },
    iconTheme: {
      primary: '#28a745',
      secondary: '#d4edda',
    },
  });
};

/**
 * Toast d'erreur
 */
export const showError = (message, duration = 4000) => {
  return toast.error(message, {
    duration,
    position: 'top-right',
    style: {
      background: '#f8d7da',
      color: '#721c24',
      border: '1px solid #f5c6cb',
      borderLeft: '4px solid #dc3545',
      padding: '16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
    },
    iconTheme: {
      primary: '#dc3545',
      secondary: '#f8d7da',
    },
  });
};

/**
 * Toast d'information
 */
export const showInfo = (message, duration = 3000) => {
  return toast(message, {
    duration,
    position: 'top-right',
    icon: 'ℹ️',
    style: {
      background: '#d1ecf1',
      color: '#0c5460',
      border: '1px solid #bee5eb',
      borderLeft: '4px solid #17a2b8',
      padding: '16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
    },
  });
};

/**
 * Toast de warning
 */
export const showWarning = (message, duration = 3500) => {
  return toast(message, {
    duration,
    position: 'top-right',
    icon: '⚠️',
    style: {
      background: '#fff3cd',
      color: '#856404',
      border: '1px solid #ffeeba',
      borderLeft: '4px solid #ffc107',
      padding: '16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
    },
  });
};

/**
 * Toast de chargement
 */
export const showLoading = (message) => {
  return toast.loading(message, {
    position: 'top-right',
    style: {
      background: '#e7f3ff',
      color: '#004085',
      border: '1px solid #b8daff',
      borderLeft: '4px solid #007bff',
      padding: '16px',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '500',
    },
  });
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
    },
    {
      position: 'top-right',
      style: {
        padding: '16px',
        borderRadius: '8px',
        fontSize: '14px',
        fontWeight: '500',
      },
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
