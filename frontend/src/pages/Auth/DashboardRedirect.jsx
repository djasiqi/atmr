import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthToken from '../../hooks/useAuthToken';

const DashboardRedirect = ({ forceDemoNamespace = false }) => {
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

      const role = (user.role || '').toLowerCase();

      // Admin: toujours vers le dashboard admin (pas de namespace demo)
      if (role === 'admin') {
        navigate(`/dashboard/admin/${user.public_id}`, { replace: true });
        return;
      }

      if (process.env.REACT_APP_DEMO_MODE === 'true') {
        navigate('/demo/home', { replace: true });
        return;
      }

      const authEnv = (localStorage.getItem('lirie_auth_env') || '').toLowerCase();
      const dashboardRoot =
        forceDemoNamespace || authEnv === 'demo' ? '/demo/dashboard' : '/dashboard';

      if (role === 'driver') {
        navigate(`${dashboardRoot}/driver/${user.public_id}`, { replace: true });
      } else if (role === 'company') {
        navigate(`${dashboardRoot}/company/${user.public_id}`, { replace: true });
      } else if (role === 'institution') {
        // ✅ ÉTAPE 6: Redirection portail Institution
        navigate(`${dashboardRoot}/institution/${user.public_id}`, { replace: true });
      } else {
        navigate(`${dashboardRoot}/${role}/${user.public_id}`, { replace: true });
      }
    }
  }, [user, navigate, forceDemoNamespace]);

  return <div>Chargement...</div>;
};

export default DashboardRedirect;
