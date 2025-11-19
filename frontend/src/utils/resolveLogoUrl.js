const RAW_API_BASE = (
  process.env.REACT_APP_API_BASE_URL ||
  process.env.REACT_APP_API_URL ||
  ''
).trim();

const API_BASE = RAW_API_BASE.replace(/\/+$/, '');

// Construire l'origine de l'API (sans /api/v1) de manière robuste
const getApiOrigin = () => {
  if (!RAW_API_BASE) {
    console.log('[resolveLogoUrl] RAW_API_BASE est vide');
    return '';
  }

  console.log('[resolveLogoUrl] RAW_API_BASE:', RAW_API_BASE);

  // Si c'est déjà une URL complète, extraire l'origine
  if (RAW_API_BASE.startsWith('http://') || RAW_API_BASE.startsWith('https://')) {
    try {
      const url = new URL(RAW_API_BASE);
      const origin = `${url.protocol}//${url.host}`;
      console.log('[resolveLogoUrl] API_ORIGIN extrait via URL:', origin);
      return origin;
    } catch (e) {
      console.warn('[resolveLogoUrl] Erreur lors de la création de URL:', e);
      // Si new URL échoue, utiliser le regex
      const origin = RAW_API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '');
      console.log('[resolveLogoUrl] API_ORIGIN extrait via regex:', origin);
      // Vérifier que l'origine est valide (contient au moins https://domain)
      if (origin.match(/^https?:\/\/[^/]+$/)) {
        return origin;
      }
      console.warn('[resolveLogoUrl] API_ORIGIN invalide après regex:', origin);
    }
  } else {
    console.log('[resolveLogoUrl] RAW_API_BASE ne commence pas par http:// ou https://');
  }

  return '';
};

const API_ORIGIN = getApiOrigin();
console.log('[resolveLogoUrl] API_ORIGIN final:', API_ORIGIN);

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

    console.log('[resolveLogoUrl] Construction URL pour uploads:', normalized);
    console.log('[resolveLogoUrl] API_ORIGIN:', API_ORIGIN);
    console.log('[resolveLogoUrl] API_BASE:', API_BASE);

    // Priorité 1: API_ORIGIN si défini et valide (doit être https://domain ou http://domain)
    if (API_ORIGIN && /^https?:\/\/[^/]+$/.test(API_ORIGIN)) {
      base = API_ORIGIN;
      console.log('[resolveLogoUrl] Utilisation API_ORIGIN:', base);
    }
    // Priorité 2: Extraire l'origine depuis API_BASE si c'est une URL complète
    else if (API_BASE && (API_BASE.startsWith('http://') || API_BASE.startsWith('https://'))) {
      try {
        const url = new URL(API_BASE);
        base = `${url.protocol}//${url.host}`;
        console.log('[resolveLogoUrl] Utilisation API_BASE (via URL):', base);
      } catch (e) {
        console.warn('[resolveLogoUrl] Erreur URL(API_BASE):', e);
        // Fallback: utiliser le regex
        const extracted = API_BASE.replace(/\/api.*$/, '').replace(/\/+$/, '');
        console.log('[resolveLogoUrl] Extraction via regex:', extracted);
        if (/^https?:\/\/[^/]+$/.test(extracted)) {
          base = extracted;
          console.log('[resolveLogoUrl] Utilisation API_BASE (via regex):', base);
        }
      }
    }
    // Priorité 3: window.location.origin (domaine actuel)
    if (!base && typeof window !== 'undefined' && window.location?.origin) {
      base = window.location.origin;
      console.log('[resolveLogoUrl] Utilisation window.location.origin:', base);
    }

    // S'assurer que base est valide avant de concaténer (doit être https://domain ou http://domain)
    if (base && /^https?:\/\/[^/]+$/.test(base)) {
      const finalUrl = `${base}${normalized}`;
      console.log('[resolveLogoUrl] URL finale construite:', finalUrl);
      return finalUrl;
    }

    // Si on arrive ici, log pour debug
    console.error('[resolveLogoUrl] ❌ Impossible de construire une URL valide pour:', normalized, {
      API_ORIGIN,
      API_BASE,
      base,
      windowOrigin: typeof window !== 'undefined' ? window.location?.origin : 'N/A',
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
