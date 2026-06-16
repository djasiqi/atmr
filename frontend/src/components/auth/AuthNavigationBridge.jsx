import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AUTH_NAVIGATE_EVENT } from '../../utils/authNavigation';

/** Relie les événements de navigation auth (logout) au routeur React. */
export default function AuthNavigationBridge() {
  const navigate = useNavigate();

  useEffect(() => {
    const handler = (event) => {
      const { to, replace = true } = event?.detail || {};
      if (!to) return;
      navigate(to, { replace });
    };
    window.addEventListener(AUTH_NAVIGATE_EVENT, handler);
    return () => window.removeEventListener(AUTH_NAVIGATE_EVENT, handler);
  }, [navigate]);

  return null;
}
