/**
 * Capacités admin.* — aligné sur GET /admin/capabilities + flag enforced.
 *
 * Mode compat (enforced=false) : l’UI autorise tout (legacy admin = accès complet).
 * Mode enforced=true : seule la liste effective compte.
 */

export const ADMIN_CAP = {
  OVERVIEW_READ: 'admin.overview.read',
  BOOKINGS_READ: 'admin.bookings.read',
  BOOKINGS_EXPORT: 'admin.bookings.export',
  PARTNERS_READ: 'admin.partners.read',
  ORGANIZATIONS_READ: 'admin.organizations.read',
  ACCOUNTS_READ: 'admin.accounts.read',
  USERS_MANAGE: 'admin.users.manage',
  USERS_SECURITY: 'admin.users.security',
  BILLING_READ: 'admin.billing.read',
  BILLING_LOCK: 'admin.billing.lock',
  BILLING_ISSUE: 'admin.billing.issue',
  BILLING_VALIDATE: 'admin.billing.validate',
  CONFIGURATION_MANAGE: 'admin.configuration.manage',
  LABS_READ: 'admin.labs.read',
  LABS_EXECUTE: 'admin.labs.execute',
};

/**
 * @param {string[]|null|undefined} list
 * @param {string} capability
 * @param {{ enforced?: boolean }} [opts]
 */
export function hasAdminCapability(list, capability, { enforced = false } = {}) {
  if (!enforced) return true;
  return Array.isArray(list) && list.includes(capability);
}
