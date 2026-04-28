/**
 * Hook pour le contexte company dashboard (localhost:3000).
 * Lit company_user + token (snake_case avec fallback camelCase pendant migration).
 */
import { useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';
import {
  clearCompanyScopedTokens,
  getActiveUser,
  getCompanyScopedAccessToken,
} from '../utils/webAuthSession';

const useCompanyAuthToken = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = getCompanyScopedAccessToken();
    const storedUser = getActiveUser();

    if (token) {
      try {
        const decoded = jwtDecode(token);
        const currentTime = Date.now() / 1000;
        if (decoded.exp && decoded.exp < currentTime) {
          clearCompanyScopedTokens();
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
        clearCompanyScopedTokens();
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
