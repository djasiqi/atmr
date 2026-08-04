/**
 * Builders d’URL admin — tout nouveau lien admin doit passer par ces helpers.
 * Préfixe : /dashboard/admin/:publicId
 */

function assertPublicId(publicId) {
  if (!publicId) {
    throw new Error('adminRoutePaths: publicId requis');
  }
  return String(publicId);
}

function join(publicId, ...segments) {
  const id = assertPublicId(publicId);
  const rest = segments
    .filter((s) => s !== undefined && s !== null && s !== '')
    .map((s) => String(s).replace(/^\/+|\/+$/g, ''))
    .filter(Boolean);
  return `/dashboard/admin/${id}${rest.length ? `/${rest.join('/')}` : ''}`;
}

export const adminPaths = {
  overview: (publicId) => join(publicId),

  operations: (publicId) => join(publicId, 'operations'),
  operationsBookings: (publicId) => join(publicId, 'operations', 'bookings'),
  operationsBooking: (publicId, bookingId) =>
    join(publicId, 'operations', 'bookings', bookingId),

  partners: (publicId) => join(publicId, 'partners'),
  partnersOrganizations: (publicId) => join(publicId, 'partners', 'organizations'),
  partnersUsers: (publicId) => join(publicId, 'partners', 'users'),
  partnersDemoRequests: (publicId) => join(publicId, 'partners', 'demo-requests'),

  finance: (publicId) => join(publicId, 'finance'),
  financeFactures: (publicId) => join(publicId, 'finance', 'factures'),
  financeReleves: (publicId) => join(publicId, 'finance', 'releves'),
  financeConfig: (publicId) => join(publicId, 'finance', 'config'),

  configuration: (publicId) => join(publicId, 'configuration'),

  advanced: (publicId) => join(publicId, 'advanced'),
  advancedPlatform: (publicId, segment = 'overview') =>
    join(publicId, 'advanced', 'platform', segment),
  advancedLabsShadowMode: (publicId) => join(publicId, 'advanced', 'labs', 'shadow-mode'),
  advancedLabsOptuna: (publicId) => join(publicId, 'advanced', 'labs', 'optuna'),
};

/**
 * Préfixe absolu admin pour un publicId (sans slash final).
 * @param {string} publicId
 */
export function adminBasePath(publicId) {
  return join(publicId);
}
