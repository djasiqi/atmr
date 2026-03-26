import { useLirieCompany } from '../hooks/useLirieCompany';

/**
 * Shell entreprise optionnel : déclenche la query profil (`lirieKeys.company`) au montage.
 * Le cache TanStack Query est partagé avec useLirieCompany / useCompanyData.
 */
export default function CompanyShellProvider({ children }) {
  useLirieCompany();
  return children;
}
