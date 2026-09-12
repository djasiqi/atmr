/**
 * Politique unique des URLs de logo (rendu <img> + persistance).
 *
 * Persisté : /uploads/... ou https://
 * Preview locale : blob: uniquement si allowPreview
 * Refusé : javascript, data, file, vbscript, http utilisateur, //protocol-relative
 */

export const LOGO_URL_DENIED_MESSAGE = 'URL de logo non autorisée';
export const MAX_LOGO_URL_LENGTH = 500;

const UPLOADS_PATH_RE = /^\/uploads\/[A-Za-z0-9._/-]+$/;

function fail() {
  return { ok: false, value: null };
}

function isSafeUploadsPath(path) {
  if (!path || path.includes('..') || path.includes('//')) {
    return false;
  }
  return UPLOADS_PATH_RE.test(path);
}

/**
 * Valide une URL destinée à être stockée (jamais blob:/data:).
 * @param {unknown} raw
 * @returns {{ ok: boolean, value: string | null }}
 */
export function validatePersistedLogoUrl(raw) {
  if (raw == null) {
    return { ok: true, value: null };
  }
  if (typeof raw !== 'string') {
    return fail();
  }
  const value = raw.trim();
  if (!value) {
    return { ok: true, value: null };
  }
  if (value.length > MAX_LOGO_URL_LENGTH) {
    return fail();
  }
  if (/[\r\n\0]/.test(value)) {
    return fail();
  }
  if (value.startsWith('//')) {
    return fail();
  }

  if (value.startsWith('/')) {
    const path = value.split('?')[0].split('#')[0];
    if (!isSafeUploadsPath(path)) {
      return fail();
    }
    return { ok: true, value: path };
  }

  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return fail();
  }
  if (parsed.protocol !== 'https:') {
    return fail();
  }
  if (parsed.username || parsed.password) {
    return fail();
  }
  if (!parsed.hostname) {
    return fail();
  }
  return {
    ok: true,
    value: `https://${parsed.host}${parsed.pathname}${parsed.search}`,
  };
}

/**
 * Autorise un blob: local pour la prévisualisation avant upload.
 * @param {unknown} raw
 * @returns {string}
 */
export function resolvePreviewBlobUrl(raw) {
  if (typeof raw !== 'string') {
    return '';
  }
  const value = raw.trim();
  if (!value.toLowerCase().startsWith('blob:')) {
    return '';
  }
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== 'blob:') {
      return '';
    }
    return parsed.href;
  } catch {
    return '';
  }
}
