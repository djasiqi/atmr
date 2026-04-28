/** Toasts / support — cohérence paiement Saferpay (réservations + dashboard). */

export function openSupportContact() {
  const path = '/contact/support';
  try {
    window.open(`${window.location.origin}${path}`, '_blank', 'noopener,noreferrer');
  } catch {
    window.location.href = path;
  }
}

/**
 * @param {typeof import('sonner').toast} toast
 * @param {Error & { httpStatus?: number }} e
 */
export function toastSaferpayCheckoutError(toast, e) {
  const status = e?.httpStatus;
  const fallback = "Le paiement en ligne n'est pas disponible pour le moment.";
  if (status === 503) {
    toast.error('Paiement en ligne temporairement indisponible', {
      description:
        "Le service de paiement n'est pas configuré ou est en maintenance. Réessayez plus tard ou contactez le support.",
      duration: 12000,
      action: {
        label: 'Contacter le support',
        onClick: openSupportContact,
      },
    });
  } else {
    toast.error(e?.message || fallback, { duration: 6000 });
  }
}
