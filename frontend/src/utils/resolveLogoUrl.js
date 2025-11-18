const RAW_API_BASE =
  (process.env.REACT_APP_API_BASE_URL || process.env.REACT_APP_API_URL || '').trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, '');
// Construire l'origine de l'API (sans /api/v1)
const API_ORIGIN = RAW_API_BASE
  ? RAW_API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '')
  : '';

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
    
    // Priorité 1: API_ORIGIN si défini et valide
    if (API_ORIGIN && API_ORIGIN.startsWith('http')) {
      base = API_ORIGIN;
    }
    // Priorité 2: window.location.origin (domaine actuel)
    else if (typeof window !== 'undefined' && window.location?.origin) {
      base = window.location.origin;
    }
    // Priorité 3: API_BASE si défini et valide
    else if (API_BASE && API_BASE.startsWith('http')) {
      // Extraire l'origine de API_BASE
      try {
        const url = new URL(API_BASE);
        base = `${url.protocol}//${url.host}`;
      } catch {
        base = API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '');
      }
    }
    
    // S'assurer que base est valide avant de concaténer
    if (base && base.startsWith('http')) {
      return `${base}${normalized}`;
    }
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

