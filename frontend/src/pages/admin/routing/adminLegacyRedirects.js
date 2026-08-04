/**
 * Redirections legacy → architecture cible.
 * Conserve search et hash. Les params de path sont mappés explicitement.
 */

import { Navigate, useLocation, useParams } from 'react-router-dom';
import { adminPaths } from './adminRoutePaths';

/**
 * @param {string} toPathname chemin absolu cible (sans search/hash)
 * @returns {JSX.Element}
 */
export function AdminLegacyRedirect({ toPathname }) {
  const location = useLocation();
  return (
    <Navigate
      to={{
        pathname: toPathname,
        search: location.search,
        hash: location.hash,
      }}
      replace
      state={location.state}
    />
  );
}

/** /reservations → operations/bookings */
export function RedirectLegacyReservations() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.operationsBookings(publicId)} />;
}

/** /reservations/:bookingId → operations/bookings/:bookingId */
export function RedirectLegacyReservationDetail() {
  const { public_id: publicId, bookingId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.operationsBooking(publicId, bookingId)} />;
}

/** /users → partners/users */
export function RedirectLegacyUsers() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.partnersUsers(publicId)} />;
}

/** /demo-requests → partners/demo-requests */
export function RedirectLegacyDemoRequests() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.partnersDemoRequests(publicId)} />;
}

/** partners → partners/organizations (conserve search/hash/state) */
export function RedirectPartnersToOrganizations() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.partnersOrganizations(publicId)} />;
}

/** /billing → finance */
export function RedirectLegacyBilling() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.finance(publicId)} />;
}

/** /billing/releves → finance/releves */
export function RedirectLegacyBillingReleves() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.financeReleves(publicId)} />;
}

/** /billing/config → finance/config */
export function RedirectLegacyBillingConfig() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.financeConfig(publicId)} />;
}

/** /settings → configuration */
export function RedirectLegacySettings() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.configuration(publicId)} />;
}

/** /shadow-mode → advanced/labs/shadow-mode */
export function RedirectLegacyShadowMode() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.advancedLabsShadowMode(publicId)} />;
}

/** /optuna → advanced/labs/optuna */
export function RedirectLegacyOptuna() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.advancedLabsOptuna(publicId)} />;
}

/** /platform-ops → advanced/platform/overview */
export function RedirectLegacyPlatformOpsIndex() {
  const { public_id: publicId } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.advancedPlatform(publicId, 'overview')} />;
}

/** /platform-ops/:segment → advanced/platform/:segment */
export function RedirectLegacyPlatformOpsSegment() {
  const { public_id: publicId, segment } = useParams();
  return <AdminLegacyRedirect toPathname={adminPaths.advancedPlatform(publicId, segment)} />;
}

/**
 * Anciennes URLs facturation / pilotage → hub Finance.
 * Conservé pour /invoices et pilotage legacy.
 */
export function RedirectToAdminFinance() {
  const { public_id: publicId } = useParams();
  const location = useLocation();
  return (
    <Navigate
      to={{
        pathname: adminPaths.finance(publicId),
        search: location.search,
        hash: location.hash,
      }}
      replace
      state={location.state}
    />
  );
}

/**
 * Table des mappings legacy (pour tests / docs).
 * Patterns relatifs sous /dashboard/admin/:publicId
 */
export const ADMIN_LEGACY_REDIRECT_SPECS = [
  { from: 'reservations', to: 'operations/bookings' },
  { from: 'reservations/:bookingId', to: 'operations/bookings/:bookingId' },
  { from: 'users', to: 'partners/users' },
  { from: 'demo-requests', to: 'partners/demo-requests' },
  { from: 'billing', to: 'finance/factures' },
  { from: 'billing/releves', to: 'finance/factures' },
  { from: 'billing/config', to: 'finance/config' },
  { from: 'settings', to: 'configuration' },
  { from: 'shadow-mode', to: 'advanced/labs/shadow-mode' },
  { from: 'optuna', to: 'advanced/labs/optuna' },
  { from: 'platform-ops', to: 'advanced/platform/overview' },
  { from: 'platform-ops/:segment', to: 'advanced/platform/:segment' },
];

/**
 * Résout un chemin relatif legacy vers un chemin relatif cible (sans publicId).
 * @param {string} relativePath ex. "reservations/abc" ou "platform-ops/runtime"
 * @returns {string|null}
 */
export function resolveLegacyRelativePath(relativePath) {
  const rel = (relativePath || '').replace(/^\/+|\/+$/g, '');
  if (!rel) return null;

  const bookingMatch = rel.match(/^reservations\/([^/]+)$/);
  if (bookingMatch) return `operations/bookings/${bookingMatch[1]}`;

  const platformMatch = rel.match(/^platform-ops(?:\/([^/]+))?$/);
  if (platformMatch) {
    return `advanced/platform/${platformMatch[1] || 'overview'}`;
  }

  const staticMap = {
    reservations: 'operations/bookings',
    users: 'partners/users',
    'demo-requests': 'partners/demo-requests',
    billing: 'finance/factures',
    'billing/releves': 'finance/factures',
    'billing/config': 'finance/config',
    'billing/pilotage': 'finance/factures',
    settings: 'configuration',
    'shadow-mode': 'advanced/labs/shadow-mode',
    optuna: 'advanced/labs/optuna',
    'platform-billing': 'finance/factures',
    invoices: 'finance/factures',
  };

  if (staticMap[rel]) return staticMap[rel];

  if (rel.startsWith('billing/pilotage/')) return 'finance/factures';
  if (rel === 'finance' || rel === 'finance/releves') return 'finance/factures';
  if (rel.startsWith('invoices/')) return 'finance';

  return null;
}
