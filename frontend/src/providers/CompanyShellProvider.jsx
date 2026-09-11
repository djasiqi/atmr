import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useLirieCompany } from '../hooks/useLirieCompany';
import { prefetchCompanyReservationsList } from '../utils/companyReservationsPrefetch';

/**
 * Shell entreprise : profil + prefetch liste Réservations si on est déjà sur la page
 * (ne pas attendre le chunk lazy pour lancer la requête critique).
 */
export default function CompanyShellProvider({ children }) {
  useLirieCompany();
  const location = useLocation();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!/\/reservations\/?$/.test(location.pathname || '')) return;
    void prefetchCompanyReservationsList(queryClient);
  }, [location.pathname, queryClient]);

  return children;
}
