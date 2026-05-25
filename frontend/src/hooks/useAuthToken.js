import { useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';
import {
  getAuthEnv,
  getEnvAccessToken,
  getEnvRefreshToken,
  getEnvUser,
  normalizeAuthRole,
  removeLegacyGlobalTokens,
} from '../utils/webAuthSession';

const useAuthToken = () => {
  const [user, setUser] = useState(null);

  // Fonction de lecture du localStorage → état user
  const readAuth = useCallback(() => {
    const env = getAuthEnv();
    const token = getEnvAccessToken(env, { allowLegacy: true });
    const storedUser = getEnvUser(env);

    // Si on a un token dans localStorage (mode mobile), le décoder
    if (token) {
      try {
        const decoded = jwtDecode(token);

        // Vérifier expiration
        const currentTime = Date.now() / 1000;
        if (decoded.exp && decoded.exp < currentTime) {
          console.warn('🔐 Token expiré');
          removeLegacyGlobalTokens();
          setUser(null);
          return;
        }
        const role = normalizeAuthRole(decoded.role);

        setUser({
          ...decoded,
          role,
          isCompany: role === 'company',
          isDriver: role === 'driver',
          isClient: role === 'client',
          companyId: decoded.company_id,
          userId: decoded.sub,
          public_id: decoded.sub,
        });
      } catch (error) {
        console.error('❌ Erreur lors du décodage du token:', error);
        removeLegacyGlobalTokens();
        setUser(null);
      }
    } else if (storedUser) {
      // Mode cookies httpOnly : utiliser les infos utilisateur stockées
      const role = normalizeAuthRole(storedUser.role);
      setUser({
        ...storedUser,
        role,
        isCompany: role === 'company',
        isDriver: role === 'driver',
        isClient: role === 'client',
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
      if (
        e.key === 'authToken' ||
        e.key === 'user' ||
        e.key === 'lirie_auth_env' ||
        e.key?.endsWith('_access_token') ||
        e.key?.endsWith('_user') ||
        e.key === null
      ) {
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
  const env = getAuthEnv();
  const token = getEnvAccessToken(env, { allowLegacy: true });
  if (!token) return null;
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    if (decoded?.exp && decoded.exp < currentTime) {
      // Token expiré => l'effacer pour basculer sur le mode cookies httpOnly si présent.
      removeLegacyGlobalTokens();
      return null;
    }
  } catch {
    // Token illisible => éviter de le réutiliser.
    removeLegacyGlobalTokens();
    return null;
  }
  return token;
}

export function getRefreshToken() {
  const env = getAuthEnv();
  const token = getEnvRefreshToken(env, { allowLegacy: true });
  if (!token) return null;
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    if (decoded?.exp && decoded.exp < currentTime) {
      removeLegacyGlobalTokens();
      return null;
    }
  } catch {
    removeLegacyGlobalTokens();
    return null;
  }
  return token;
}
