/**
 * Événements analytics SEO (sans données santé / trajet / identité patient).
 * Branche un collector optionnel ; no-op par défaut.
 */

const ALLOWED_EVENTS = new Set([
  'seo_public_page_view',
  'seo_contact_click',
  'seo_demo_request',
  'seo_institution_cta',
  'seo_transport_company_cta',
  'seo_patient_booking_start',
  'seo_phone_click',
  'seo_email_click',
]);

const SENSITIVE_KEYS = /patient|booking_id|address|lat|lon|token|email|phone|name/i;

/**
 * @param {string} eventName
 * @param {Record<string, string | number | boolean | undefined | null>} [payload]
 */
export function trackSeoEvent(eventName, payload = {}) {
  if (!ALLOWED_EVENTS.has(eventName)) {
    if (process.env.NODE_ENV === 'development') {
      console.warn('[seo-analytics] événement non autorisé:', eventName);
    }
    return;
  }

  const safe = {
    landing_page: typeof window !== 'undefined' ? window.location.pathname : undefined,
    referrer: typeof document !== 'undefined' ? document.referrer || undefined : undefined,
    device_type:
      typeof window !== 'undefined' && window.matchMedia?.('(max-width: 768px)')?.matches
        ? 'mobile'
        : 'desktop',
  };

  const params = new URLSearchParams(
    typeof window !== 'undefined' ? window.location.search : ''
  );
  for (const key of ['utm_source', 'utm_medium', 'utm_campaign']) {
    const value = params.get(key);
    if (value) safe[key] = value;
  }

  for (const [key, value] of Object.entries(payload || {})) {
    if (SENSITIVE_KEYS.test(key)) continue;
    if (value == null) continue;
    if (typeof value === 'string' && /@|\d{6,}/.test(value)) continue;
    safe[key] = value;
  }

  try {
    if (typeof window !== 'undefined' && typeof window.__LIRIE_SEO_ANALYTICS__ === 'function') {
      window.__LIRIE_SEO_ANALYTICS__(eventName, safe);
      return;
    }
  } catch (_) {
    /* ignore */
  }

  if (process.env.NODE_ENV === 'development') {
    // eslint-disable-next-line no-console
    console.info('[seo-analytics]', eventName, safe);
  }
}

/**
 * Hook léger pour vue page publique (à appeler une fois au mount).
 * @param {string} landingPage
 */
export function trackPublicSeoPageView(landingPage) {
  trackSeoEvent('seo_public_page_view', { landing_page: landingPage });
}
