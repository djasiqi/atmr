import { useQuery } from '@tanstack/react-query';

import { featureFlags } from '@/services/featureFlags';
import { getInstitutionMe } from '@/services/institutionApi';
import { queryKeys } from '@/services/queryKeys';

type InstitutionRole =
  | 'institution_admin'
  | 'institution_requester'
  | 'institution_reader'
  | 'institution_billing'
  | 'institution_curator'
  | '';

function normalizeRole(value: string | null | undefined): InstitutionRole {
  const role = String(value ?? '').trim().toLowerCase();
  if (
    role === 'institution_admin'
    || role === 'institution_requester'
    || role === 'institution_reader'
    || role === 'institution_billing'
    || role === 'institution_curator'
  ) {
    return role;
  }
  return '';
}

export function useInstitutionPermissions() {
  const meQuery = useQuery({
    queryKey: queryKeys.institutionMe,
    queryFn: getInstitutionMe,
  });

  const role = normalizeRole(meQuery.data?.institution_role);
  const strictRoleGuards = featureFlags.institutionMobileRoleGuardsEnabled;
  const canCreateRequest = !strictRoleGuards ? true : (
    role === 'institution_admin'
    || role === 'institution_requester'
    || role === 'institution_curator'
  );
  const canSendRequest = canCreateRequest;
  const canCreatePatient = !strictRoleGuards ? true : canCreateRequest;
  const canEditNotifications = !strictRoleGuards
    ? true
    : role === 'institution_admin' || role === 'institution_billing';

  return {
    role,
    canCreateRequest,
    canSendRequest,
    canCreatePatient,
    canEditNotifications,
    isReader: role === 'institution_reader',
    isBilling: role === 'institution_billing',
    isLoading: meQuery.isLoading,
  };
}
