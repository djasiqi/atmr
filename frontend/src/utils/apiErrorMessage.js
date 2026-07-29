import { getServiceUnavailableMessage } from '../constants/platformSupport';

const SERVICE_UNAVAILABLE_HTTP_STATUSES = new Set([404, 502, 503, 504]);
const SERVICE_UNAVAILABLE_NETWORK_CODES = new Set([
  'ERR_NETWORK',
  'ECONNABORTED',
  'ETIMEDOUT',
  'ECONNREFUSED',
]);
const SERVICE_UNAVAILABLE_MESSAGE_PATTERNS = [
  /404\s+page\s+not\s+found/i,
  /\bpage\s+not\s+found\b/i,
  /gateway\s+timeout/i,
  /service\s+temporairement\s+indisponible/i,
  /backend\s+.*non\s+accessible/i,
  /network\s+error/i,
  /proxy\s+error/i,
];

const _extractBackendMessage = (error) => {
  const d = error?.response?.data;
  if (typeof d === 'string' && d.trim()) {
    return d.trim();
  }
  if (!d || typeof d !== 'object') {
    return null;
  }
  const candidates = [d.message, d.detail, d.error];
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate.trim();
    }
  }
  return null;
};

const _matchesServiceUnavailableMessage = (value) => {
  const text = String(value || '').trim();
  if (!text) return false;
  return SERVICE_UNAVAILABLE_MESSAGE_PATTERNS.some((pattern) => pattern.test(text));
};

/**
 * Indique si l'erreur correspond à une indisponibilité serveur (maintenance, panne, proxy).
 * @param {unknown} error
 * @returns {boolean}
 */
export function isServiceUnavailableError(error) {
  const status = error?.response?.status;
  if (SERVICE_UNAVAILABLE_HTTP_STATUSES.has(status)) {
    return true;
  }

  const code = String(error?.code || '').trim();
  if (code && SERVICE_UNAVAILABLE_NETWORK_CODES.has(code)) {
    return true;
  }

  const backendMessage = _extractBackendMessage(error);
  if (_matchesServiceUnavailableMessage(backendMessage)) {
    return true;
  }

  const axiosMessage = typeof error?.message === 'string' ? error.message.trim() : '';
  if (!error?.response) {
    if (_matchesServiceUnavailableMessage(axiosMessage)) {
      return true;
    }
    if (/^Request failed with status code (404|502|503|504)$/i.test(axiosMessage)) {
      return true;
    }
  }

  return false;
}

/**
 * Extrait un message utilisateur lisible depuis une erreur Axios / API LIRIE.
 * @param {unknown} error
 * @param {string} [fallback]
 * @returns {string}
 */
export function getApiErrorMessage(error, fallback = 'Une erreur est survenue.') {
  if (isServiceUnavailableError(error)) {
    return getServiceUnavailableMessage();
  }
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
    const msg = d.message.trim();
    // Marshmallow brut : préciser le champ si disponible
    if (/^Missing data for required field\.?$/i.test(msg)) {
      const fields = d.details?.fields || d.details?.errors || d.errors;
      if (fields && typeof fields === 'object') {
        const fieldName = Object.keys(fields)[0];
        if (fieldName) {
          const labels = {
            email: 'email',
            phone: 'téléphone',
            password: 'mot de passe',
            username: "nom d'utilisateur",
            first_name: 'prénom',
            last_name: 'nom',
          };
          const label = labels[fieldName] || fieldName;
          return `Champ obligatoire manquant : ${label}.`;
        }
      }
      return 'Un champ obligatoire est manquant.';
    }
    return msg;
  }

  // Réponses legacy : message utilisateur dans `error` + `error_code` (ex. validation 400)
  if (
    typeof d.error_code === 'string' &&
    d.error_code.trim() &&
    typeof d.error === 'string' &&
    d.error.trim()
  ) {
    return d.error.trim();
  }

  const inner = d.data;
  if (inner && typeof inner === 'object' && typeof inner.message === 'string' && inner.message.trim()) {
    return inner.message.trim();
  }

  if (typeof d.error === 'string' && d.error.trim()) {
    const code = d.error.trim();
    const known = {
      payment_unavailable: 'Le paiement en ligne est temporairement indisponible.',
      saferpay_configuration: 'Configuration du prestataire de paiement (Saferpay) incomplète.',
      saferpay_initialize_failed:
        'Impossible de démarrer le paiement Saferpay (refus ou configuration). Réessayez plus tard ou contactez le support.',
      validation_error: typeof d.message === 'string' ? d.message : null,
    };
    if (known[code]) {
      return known[code];
    }
    /* Message utilisateur direct (ex. "Client introuvable pour cette entreprise") */
    return code;
  }

  return fallback;
}
