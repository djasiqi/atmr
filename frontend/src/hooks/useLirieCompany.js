import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchCompanyInfo } from '../services/companyService';
import { joinCompanyRoom } from '../services/companySocket';
import { getAccessToken } from './useAuthToken';
import { lirieKeys } from '../queryKeys/lirie';

/**
 * Profil entreprise uniquement (TanStack Query + room Socket.IO).
 * Ne charge pas réservations ni chauffeurs — utiliser useCompanyData pour l’opérationnel.
 */
export function useLirieCompany() {
  const {
    data: company,
    isLoading,
    isFetching,
    error: companyQueryError,
    refetch: reloadCompany,
  } = useQuery({
    queryKey: lirieKeys.company(),
    queryFn: async () => {
      const token = getAccessToken();
      const hasToken = !!token;
      const hasUser = typeof localStorage !== 'undefined' && !!localStorage.getItem('user');
      if (!hasToken && !hasUser) {
        throw new Error('AUTH_REQUIRED');
      }
      const data = await fetchCompanyInfo();
      if (data?.error === true) {
        throw new Error('COMPANY_LOAD_FAILED');
      }
      return data;
    },
    staleTime: 60_000,
    retry: 1,
  });

  useEffect(() => {
    const cid = company?.id;
    if (cid && Number(cid) > 0) {
      Promise.resolve(joinCompanyRoom(Number(cid))).catch(() => {});
    }
  }, [company?.id]);

  const companyError = useMemo(() => {
    if (!companyQueryError) return null;
    if (companyQueryError.message === 'AUTH_REQUIRED') {
      return 'Authentification manquante. Veuillez vous reconnecter.';
    }
    return "Erreur lors du chargement de l'entreprise.";
  }, [companyQueryError]);

  return {
    company: company ?? null,
    loadingCompany: isLoading || isFetching,
    companyError,
    reloadCompany,
  };
}
