import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthToken from '../../hooks/useAuthToken';

const DashboardRedirect = () => {
  const user = useAuthToken();
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      // ⚡ Vérifier que public_id existe avant de naviguer
      if (!user.public_id) {
        console.error('❌ public_id manquant dans le token, redirection vers login');
        navigate('/login', { replace: true });
        return;
      }

      // Normaliser le rôle en minuscules pour la comparaison
      const role = (user.role || '').toLowerCase();

      if (role === 'driver') {
        navigate(`/dashboard/driver/${user.public_id}`, { replace: true });
      } else if (role === 'company') {
        navigate(`/dashboard/company/${user.public_id}`, { replace: true });
      } else if (role === 'institution') {
        // ✅ ÉTAPE 6: Redirection portail Institution
        navigate(`/dashboard/institution/${user.public_id}`, { replace: true });
      } else {
        navigate(`/dashboard/${role}/${user.public_id}`, { replace: true });
      }
    }
  }, [user, navigate]);

  return <div>Chargement...</div>;
};

export default DashboardRedirect;
