const RAW_API_BASE =
  (process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '').trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, '');

// Construire l'origine de l'API (sans /api/v1) de manière robuste
const getApiOrigin = () => {
  if (!RAW_API_BASE) return '';
  
  // Si c'est déjà une URL complète, extraire l'origine
  if (RAW_API_BASE.startsWith('http://') || RAW_API_BASE.startsWith('https://')) {
    try {
      const url = new URL(RAW_API_BASE);
      return `${url.protocol}//${url.host}`;
    } catch {
      // Si new URL échoue, utiliser le regex
      const origin = RAW_API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '');
      // Vérifier que l'origine est valide (contient au moins https://domain)
      if (origin.match(/^https?:\/\/[^/]+$/)) {
        return origin;
      }
    }
  }
  
  return '';
};

const API_ORIGIN = getApiOrigin();

export const resolveLogoUrl = (value) => {
  if (!value) {
    return '';
  }

  const str = String(value).trim();
  
  // Si c'est déjà une URL complète (http/https/data/blob), la retourner telle quelle
  if (/^(https?:|data:|blob:)/i.test(str)) {
    return str;
  }

  // Normaliser le chemin (s'assurer qu'il commence par /)
  const normalized = str.startsWith('/') ? str : `/${str}`;

  // Pour les fichiers uploads, utiliser l'origine de l'API ou l'origine du navigateur
  if (normalized.startsWith('/uploads/')) {
    let base = '';
    
    // Priorité 1: API_ORIGIN si défini et valide (doit être https://domain ou http://domain)
    if (API_ORIGIN && /^https?:\/\/[^/]+$/.test(API_ORIGIN)) {
      base = API_ORIGIN;
    }
    // Priorité 2: Extraire l'origine depuis API_BASE si c'est une URL complète
    else if (API_BASE && (API_BASE.startsWith('http://') || API_BASE.startsWith('https://'))) {
      try {
        const url = new URL(API_BASE);
        base = `${url.protocol}//${url.host}`;
      } catch {
        // Fallback: utiliser le regex
        const extracted = API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '');
        if (/^https?:\/\/[^/]+$/.test(extracted)) {
          base = extracted;
        }
      }
    }
    // Priorité 3: window.location.origin (domaine actuel)
    if (!base && typeof window !== 'undefined' && window.location?.origin) {
      base = window.location.origin;
    }
    
    // S'assurer que base est valide avant de concaténer (doit être https://domain ou http://domain)
    if (base && /^https?:\/\/[^/]+$/.test(base)) {
      return `${base}${normalized}`;
    }
    
    // Si on arrive ici, log pour debug
    console.warn('[resolveLogoUrl] Impossible de construire une URL valide pour:', normalized, {
      API_ORIGIN,
      API_BASE,
      windowOrigin: typeof window !== 'undefined' ? window.location?.origin : 'N/A'
    });
  }

  // Pour les autres chemins, utiliser API_BASE
  if (API_BASE) {
    return `${API_BASE}${normalized}`;
  }

  // Fallback: utiliser l'origine du navigateur
  if (typeof window !== 'undefined' && window.location?.origin) {
    return `${window.location.origin}${normalized}`;
  }

  return normalized;
};

export default resolveLogoUrl;

