import {
  resolvePreviewBlobUrl,
  validatePersistedLogoUrl,
} from './logoUrlPolicy';

const RAW_API_BASE = (
  process.env.REACT_APP_API_BASE_URL ||
  process.env.REACT_APP_API_URL ||
  ''
).trim();

const RAW_SOCKET_URL = (process.env.REACT_APP_SOCKET_URL || '').trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, '');

const getHttpOrigin = (raw) => {
  if (!raw) {
    return '';
  }
  if (!raw.startsWith('http://') && !raw.startsWith('https://')) {
    return '';
  }
  try {
    const url = new URL(raw);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return '';
    }
    return `${url.protocol}//${url.host}`;
  } catch {
    return '';
  }
};

const API_ORIGIN = getHttpOrigin(RAW_API_BASE);
const SOCKET_ORIGIN = getHttpOrigin(RAW_SOCKET_URL);

function joinUploadsPath(uploadsPath) {
  if (API_ORIGIN) {
    return `${API_ORIGIN}${uploadsPath}`;
  }
  if (API_BASE && (API_BASE.startsWith('http://') || API_BASE.startsWith('https://'))) {
    const origin = getHttpOrigin(API_BASE);
    if (origin) {
      return `${origin}${uploadsPath}`;
    }
  }
  if (SOCKET_ORIGIN) {
    return `${SOCKET_ORIGIN}${uploadsPath}`;
  }
  if (typeof window !== 'undefined' && window.location?.origin) {
    return `${window.location.origin}${uploadsPath}`;
  }
  return uploadsPath;
}

/**
 * Résout une URL de logo pour un <img src>.
 * @param {unknown} value
 * @param {{ allowPreview?: boolean }} [options]
 * @returns {string} URL sûre ou chaîne vide (fallback visuel)
 */
export const resolveLogoUrl = (value, options = {}) => {
  if (options.allowPreview) {
    const blobUrl = resolvePreviewBlobUrl(value);
    if (blobUrl) {
      return blobUrl;
    }
  }

  const persisted = validatePersistedLogoUrl(value);
  if (!persisted.ok || !persisted.value) {
    return '';
  }

  if (persisted.value.startsWith('/uploads/')) {
    return joinUploadsPath(persisted.value);
  }

  return persisted.value;
};

export default resolveLogoUrl;
