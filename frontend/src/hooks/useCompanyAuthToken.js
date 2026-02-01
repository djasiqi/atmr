/**
 * Hook pour le contexte company dashboard (localhost:3000).
 * Lit company_user + token (snake_case avec fallback camelCase pendant migration).
 */
import { useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';

const COMPANY_USER_KEY = 'company_user';
const COMPANY_ACCESS_TOKEN_KEY = 'company_access_token';
const COMPANY_ACCESS_TOKEN_LEGACY = 'company_authToken';
const COMPANY_REFRESH_KEY = 'company_refresh_token';
const COMPANY_REFRESH_LEGACY = 'company_refreshToken';

const getCompanyToken = () =>
  localStorage.getItem(COMPANY_ACCESS_TOKEN_KEY) || localStorage.getItem(COMPANY_ACCESS_TOKEN_LEGACY);

const useCompanyAuthToken = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = getCompanyToken();
    const rawUser = localStorage.getItem(COMPANY_USER_KEY);
    const storedUser = rawUser ? JSON.parse(rawUser) : null;

    if (token) {
      try {
        const decoded = jwtDecode(token);
        const currentTime = Date.now() / 1000;
        if (decoded.exp && decoded.exp < currentTime) {
          try {
            localStorage.removeItem(COMPANY_ACCESS_TOKEN_KEY);
            localStorage.removeItem(COMPANY_ACCESS_TOKEN_LEGACY);
            localStorage.removeItem(COMPANY_REFRESH_KEY);
            localStorage.removeItem(COMPANY_REFRESH_LEGACY);
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
        console.error('❌ useCompanyAuthToken: token invalide', error);
        try {
          localStorage.removeItem(COMPANY_ACCESS_TOKEN_KEY);
          localStorage.removeItem(COMPANY_ACCESS_TOKEN_LEGACY);
          localStorage.removeItem(COMPANY_REFRESH_KEY);
          localStorage.removeItem(COMPANY_REFRESH_LEGACY);
        } catch {}
        setUser(null);
      }
    } else if (storedUser) {
      // ✅ Garde P0 : Ne pas retourner user sans token valide.
      // company_dispatch/* exige company_access_token — sans token → 401 systématique.
      // Retourner null évite les appels prématurés / mauvais timing.
      setUser(null);
    } else {
      setUser(null);
    }
  }, []);

  return { user, isCompanyAuthReady: !!user };
};

export default useCompanyAuthToken;
