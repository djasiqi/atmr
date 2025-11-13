const RAW_API_BASE =
  (process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '').trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, '');
const API_ORIGIN = RAW_API_BASE
  ? RAW_API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '')
  : '';

export const resolveLogoUrl = (value) => {
  if (!value) {
    return '';
  }

  const str = String(value).trim();
  if (/^(https?:|data:|blob:)/i.test(str)) {
    return str;
  }

  const normalized = str.startsWith('/') ? str : `/${str}`;

  if (normalized.startsWith('/uploads/')) {
    const base =
      API_ORIGIN ||
      (typeof window !== 'undefined' && `${window.location.origin}`) ||
      '';
    return `${base}${normalized}`;
  }

  if (API_BASE) {
    return `${API_BASE}${normalized}`;
  }

  if (API_ORIGIN) {
    return `${API_ORIGIN}${normalized}`;
  }

  if (typeof window !== 'undefined' && window.location?.origin) {
    return `${window.location.origin}${normalized}`;
  }

  return normalized;
};

export default resolveLogoUrl;

