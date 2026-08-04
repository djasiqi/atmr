import { useQuery } from '@tanstack/react-query';
import { fetchAdminCapabilities } from '../services/adminService';
import { ADMIN_CAP, hasAdminCapability } from '../pages/admin/capabilities/adminCapabilities';

/**
 * Hook capacités admin.* (PR2bis).
 * Ne crée pas de second système parallèle à usePlatformCapabilities.
 */
export function useAdminCapabilities() {
  const query = useQuery({
    queryKey: ['admin', 'capabilities'],
    queryFn: fetchAdminCapabilities,
    staleTime: 3 * 60 * 1000,
    retry: 1,
  });

  const list = query.data?.capabilities_effective ?? null;
  const enforced = Boolean(query.data?.enforced);

  const can = (capability) => hasAdminCapability(list, capability, { enforced });

  return {
    ...query,
    enforced,
    capabilitiesEffective: list,
    can,
    canLabsRead: can(ADMIN_CAP.LABS_READ),
    canLabsExecute: can(ADMIN_CAP.LABS_EXECUTE),
    canBillingLock: can(ADMIN_CAP.BILLING_LOCK),
    canBillingIssue: can(ADMIN_CAP.BILLING_ISSUE),
    canBillingValidate: can(ADMIN_CAP.BILLING_VALIDATE),
    canBillingSend: can(ADMIN_CAP.BILLING_SEND),
    canBillingPayment: can(ADMIN_CAP.BILLING_PAYMENT),
    canBillingDueDate: can(ADMIN_CAP.BILLING_DUE_DATE),
    canBillingCancel: can(ADMIN_CAP.BILLING_CANCEL),
    canBillingCredit: can(ADMIN_CAP.BILLING_CREDIT),
    canBillingRead: can(ADMIN_CAP.BILLING_READ),
    canConfigurationManage: can(ADMIN_CAP.CONFIGURATION_MANAGE),
    canUsersManage: can(ADMIN_CAP.USERS_MANAGE),
    canUsersSecurity: can(ADMIN_CAP.USERS_SECURITY),
    canOrganizationsRead: can(ADMIN_CAP.ORGANIZATIONS_READ),
    canAccountsRead: can(ADMIN_CAP.ACCOUNTS_READ),
  };
}

export { ADMIN_CAP };
