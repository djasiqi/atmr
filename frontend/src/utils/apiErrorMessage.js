/**
 * Extrait un message utilisateur lisible depuis une erreur Axios / API LIRIE.
 * @param {unknown} error
 * @param {string} [fallback]
 * @returns {string}
 */
export function getApiErrorMessage(error, fallback = 'Une erreur est survenue.') {
  if (error && typeof error === 'object' && 'message' in error) {
    const m = error.message;
    if (typeof m === 'string' && m.trim() && !/^Request failed with status code \d+$/i.test(m)) {
      return m.trim();
    }
  }

  const res = error?.response;
  const d = res?.data;
  if (!d || typeof d !== 'object') {
    return fallback;
  }

  if (typeof d.message === 'string' && d.message.trim()) {
    return d.message.trim();
  }

  const inner = d.data;
  if (inner && typeof inner === 'object' && typeof inner.message === 'string' && inner.message.trim()) {
    return inner.message.trim();
  }

  if (typeof d.error === 'string' && d.error.trim()) {
    const code = d.error.trim();
    const known = {
      payment_unavailable: 'Le paiement en ligne est temporairement indisponible.',
      worldline_configuration: 'Configuration du prestataire de paiement incomplète.',
      validation_error: typeof d.message === 'string' ? d.message : null,
    };
    if (known[code]) {
      return known[code];
    }
  }

  return fallback;
}
