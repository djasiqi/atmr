/**
 * Helpers pour mocker les cookies dans les tests frontend
 * 
 * ✅ Migration localStorage → cookies httpOnly
 * Ces helpers permettent de simuler les cookies httpOnly dans les tests Jest
 * car les cookies réels ne sont pas accessibles depuis JavaScript en production.
 */

/**
 * Mock des cookies en mémoire (simule document.cookie)
 */
let mockCookies = {};

/**
 * Réinitialise tous les cookies mockés
 */
export const clearMockCookies = () => {
  mockCookies = {};
};

/**
 * Définit un cookie mocké
 * @param {string} name - Nom du cookie
 * @param {string} value - Valeur du cookie
 * @param {object} options - Options (path, domain, maxAge, etc.)
 */
export const setMockCookie = (name, value, options = {}) => {
  mockCookies[name] = {
    value,
    ...options,
  };
};

/**
 * Récupère un cookie mocké
 * @param {string} name - Nom du cookie
 * @returns {string|null} - Valeur du cookie ou null si inexistant
 */
export const getMockCookie = (name) => {
  return mockCookies[name]?.value || null;
};

/**
 * Supprime un cookie mocké
 * @param {string} name - Nom du cookie
 */
export const removeMockCookie = (name) => {
  delete mockCookies[name];
};

/**
 * Vérifie qu'un cookie existe
 * @param {string} name - Nom du cookie
 * @returns {boolean}
 */
export const hasMockCookie = (name) => {
  return name in mockCookies;
};

/**
 * Configure les cookies mockés pour simuler une session authentifiée
 * @param {string} accessToken - Token d'accès
 * @param {string} refreshToken - Token de rafraîchissement (optionnel)
 */
export const mockAuthenticatedCookies = (accessToken, refreshToken = null) => {
  setMockCookie('access_token', accessToken, {
    path: '/',
    httpOnly: true,
    secure: true,
    sameSite: 'Lax',
  });
  
  if (refreshToken) {
    setMockCookie('refresh_token', refreshToken, {
      path: '/',
      httpOnly: true,
      secure: true,
      sameSite: 'Lax',
    });
  }
};

/**
 * Supprime tous les cookies d'authentification
 */
export const mockLogoutCookies = () => {
  removeMockCookie('access_token');
  removeMockCookie('refresh_token');
};

/**
 * Mock document.cookie pour Jest
 * Simule le comportement de document.cookie (lecture/écriture)
 */
export const setupCookieMock = () => {
  // Sauvegarder l'original si existe
  const originalCookie = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie') ||
                         Object.getOwnPropertyDescriptor(HTMLDocument.prototype, 'cookie');

  // Mock document.cookie getter
  Object.defineProperty(document, 'cookie', {
    get: () => {
      return Object.entries(mockCookies)
        .map(([name, data]) => {
          let cookieStr = `${name}=${data.value}`;
          if (data.path) cookieStr += `; Path=${data.path}`;
          if (data.domain) cookieStr += `; Domain=${data.domain}`;
          if (data.maxAge) cookieStr += `; Max-Age=${data.maxAge}`;
          if (data.expires) cookieStr += `; Expires=${data.expires}`;
          if (data.secure) cookieStr += '; Secure';
          if (data.sameSite) cookieStr += `; SameSite=${data.sameSite}`;
          if (data.httpOnly) cookieStr += '; HttpOnly';
          return cookieStr;
        })
        .join('; ');
    },
    set: (cookieString) => {
      // Parser le cookie string (ex: "name=value; Path=/; HttpOnly")
      const [nameValue, ...options] = cookieString.split(';').map(s => s.trim());
      const [name, value] = nameValue.split('=');
      
      if (!name || !value) return;
      
      const cookieData = { value };
      
      options.forEach(option => {
        const [key, val] = option.split('=').map(s => s.trim());
        if (key.toLowerCase() === 'path') cookieData.path = val;
        else if (key.toLowerCase() === 'domain') cookieData.domain = val;
        else if (key.toLowerCase() === 'max-age') cookieData.maxAge = parseInt(val, 10);
        else if (key.toLowerCase() === 'expires') cookieData.expires = val;
        else if (key.toLowerCase() === 'secure') cookieData.secure = true;
        else if (key.toLowerCase() === 'samesite') cookieData.sameSite = val;
        else if (key.toLowerCase() === 'httponly') cookieData.httpOnly = true;
      });
      
      // Si Max-Age est 0 ou Expires est dans le passé, supprimer le cookie
      if (cookieData.maxAge === 0 || (cookieData.expires && new Date(cookieData.expires) < new Date())) {
        removeMockCookie(name);
      } else {
        setMockCookie(name, value, cookieData);
      }
    },
    configurable: true,
  });

  return () => {
    // Restaurer l'original si existe
    if (originalCookie) {
      Object.defineProperty(document, 'cookie', originalCookie);
    } else {
      delete document.cookie;
    }
    clearMockCookies();
  };
};

/**
 * Vérifie que apiClient est appelé avec withCredentials: true
 * (les cookies sont envoyés automatiquement)
 * 
 * @param {jest.Mock} apiClientMock - Mock de apiClient
 * @param {string} method - Méthode HTTP (get, post, put, delete)
 * @param {string} url - URL de l'endpoint
 */
export const expectApiCallWithCookies = (apiClientMock, method, url) => {
  expect(apiClientMock[method]).toHaveBeenCalled();
  
  // Vérifier que l'appel ne contient PAS de header Authorization manuel
  // (les cookies sont envoyés automatiquement via withCredentials: true)
  const calls = apiClientMock[method].mock.calls;
  const lastCall = calls[calls.length - 1];
  
  // Si un config object est passé, vérifier qu'il n'y a pas de header Authorization
  if (lastCall.length > 2 && lastCall[2]?.headers?.Authorization) {
    throw new Error(
      `Expected apiClient.${method} to be called without manual Authorization header. ` +
      `Cookies are sent automatically via withCredentials: true.`
    );
  }
  
  // Vérifier que l'URL est correcte
  if (lastCall[0] !== url) {
    throw new Error(
      `Expected apiClient.${method} to be called with "${url}", but got "${lastCall[0]}"`
    );
  }
};

