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

      if (user.role === 'driver') {
        navigate(`/dashboard/driver/${user.public_id}`, { replace: true });
      } else if (user.role === 'company') {
        navigate(`/dashboard/company/${user.public_id}`, { replace: true });
      } else {
        navigate(`/dashboard/${user.role}/${user.public_id}`, { replace: true });
      }
    }
  }, [user, navigate]);

  return <div>Chargement...</div>;
};

export default DashboardRedirect;
