import { useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';

const useAuthToken = () => {
  const [user, setUser] = useState(null);

  // Fonction de lecture du localStorage → état user
  const readAuth = useCallback(() => {
    const token = localStorage.getItem('authToken');
    const rawUser = localStorage.getItem('user');
    const storedUser = rawUser ? JSON.parse(rawUser) : null;

    // Si on a un token dans localStorage (mode mobile), le décoder
    if (token) {
      try {
        const decoded = jwtDecode(token);

        // Vérifier expiration
        const currentTime = Date.now() / 1000;
        if (decoded.exp && decoded.exp < currentTime) {
          console.warn('Token expiré');
          try {
            localStorage.removeItem('authToken');
            localStorage.removeItem('refreshToken');
          } catch {}
          setUser(null);
          return;
        }

        setUser({
          ...decoded,
          isCompany: decoded.role === 'company',
          isDriver: decoded.role === 'driver',
          isClient: decoded.role === 'client',
          companyId: decoded.company_id,
          userId: decoded.sub,
          public_id: decoded.sub,
        });
      } catch (error) {
        console.error('Erreur lors du décodage du token:', error);
        try {
          localStorage.removeItem('authToken');
          localStorage.removeItem('refreshToken');
        } catch {}
        setUser(null);
      }
    } else if (storedUser) {
      // Mode cookies httpOnly : utiliser les infos utilisateur stockées
      setUser({
        ...storedUser,
        isCompany: String(storedUser.role || '').toLowerCase() === 'company',
        isDriver: String(storedUser.role || '').toLowerCase() === 'driver',
        isClient: String(storedUser.role || '').toLowerCase() === 'client',
        companyId: storedUser.company_id,
        userId: storedUser.id,
        public_id: storedUser.public_id,
      });
    } else {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    // Lecture initiale
    readAuth();

    // Écouter les changements localStorage venant d'un autre onglet
    const handleStorageChange = (e) => {
      if (e.key === 'authToken' || e.key === 'user' || e.key === null) {
        readAuth();
      }
    };
    window.addEventListener('storage', handleStorageChange);

    // Écouter un événement custom pour les changements dans le MÊME onglet
    // (l'événement 'storage' natif ne se déclenche pas dans l'onglet qui fait le changement)
    const handleAuthChange = () => readAuth();
    window.addEventListener('auth-changed', handleAuthChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('auth-changed', handleAuthChange);
    };
  }, [readAuth]);

  return user;
};

export default useAuthToken;

// ✅ Fonction d'accès directe au token brut
export function getAccessToken() {
  const token = localStorage.getItem('authToken');
  if (!token) return null;
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    if (decoded?.exp && decoded.exp < currentTime) {
      // Token expiré => l'effacer pour basculer sur le mode cookies httpOnly si présent.
      try {
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
      } catch {}
      return null;
    }
  } catch {
    // Token illisible => éviter de le réutiliser.
    try {
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
    } catch {}
    return null;
  }
  return token;
}

export function getRefreshToken() {
  const token = localStorage.getItem('refreshToken');
  if (!token) return null;
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    if (decoded?.exp && decoded.exp < currentTime) {
      try {
        localStorage.removeItem('refreshToken');
      } catch {}
      return null;
    }
  } catch {
    try {
      localStorage.removeItem('refreshToken');
    } catch {}
    return null;
  }
  return token;
}
