/**
 * Capacités admin.* (PR2bis) — aligné sur GET /admin/capabilities.
 * Avec ADMIN_CAPABILITIES_ENFORCED=false, le backend autorise toujours (compat)
 * mais journalise les « aurait refusé » ; le front peut masquer les labs / actions.
 */

export const ADMIN_CAP = {
  OVERVIEW_READ: 'admin.overview.read',
  BOOKINGS_READ: 'admin.bookings.read',
  BOOKINGS_EXPORT: 'admin.bookings.export',
  PARTNERS_READ: 'admin.partners.read',
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
 * @param {{ enforced?: boolean }} [opts] réservé (API d’enforcement côté serveur)
 */
export function hasAdminCapability(list, capability, _opts = {}) {
  if (!list || !Array.isArray(list) || list.length === 0) {
    // Pas encore chargé / legacy : autoriser (ne pas casser l’UI)
    return true;
  }
  return list.includes(capability);
}
