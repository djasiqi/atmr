import { useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';

const useAuthToken = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('authToken');

    if (token) {
      try {
        const decoded = jwtDecode(token);

        // Vérifier expiration
        const currentTime = Date.now() / 1000;
        if (decoded.exp && decoded.exp < currentTime) {
          console.warn('🔐 Token expiré');
          setUser(null);
          return;
        }

        // Ajouter des infos structurées
        setUser({
          ...decoded,
          isCompany: decoded.role === 'company',
          isDriver: decoded.role === 'driver',
          isClient: decoded.role === 'client',
          companyId: decoded.company_id,
          userId: decoded.sub,
          public_id: decoded.sub, // ✅ Le backend envoie public_id dans le champ 'sub'
        });
      } catch (error) {
        console.error('❌ Erreur lors du décodage du token:', error);
        setUser(null);
      }
    } else {
      setUser(null);
    }
  }, []);

  return user;
};

export default useAuthToken;

// ✅ Fonction d'accès directe au token brut
export function getAccessToken() {
  return localStorage.getItem('authToken');
}

export function getRefreshToken() {
  return localStorage.getItem('refreshToken');
}
