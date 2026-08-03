/**
 * Registre de navigation admin — architecture cible uniquement (pas de legacy).
 * Les enfants Platform Ops portent `platformCapability` (segments usePlatformCapabilities).
 */

/** @typedef {{ id: string, label: string, path: string, end?: boolean, platformCapability?: string }} AdminNavChild */
/** @typedef {{ id: string, label: string, path: string, matchPrefixes: string[], children?: AdminNavChild[] }} AdminWorkspace */

/** @type {AdminWorkspace[]} */
export const ADMIN_WORKSPACES = [
  {
    id: 'overview',
    label: 'Vue d’ensemble',
    path: '',
    matchPrefixes: [''],
    children: [],
  },
  {
    id: 'operations',
    label: 'Opérations',
    path: 'operations',
    matchPrefixes: ['operations'],
    children: [
      { id: 'bookings', label: 'Transports', path: 'operations/bookings', end: false },
    ],
  },
  {
    id: 'partners',
    label: 'Partenaires',
    path: 'partners',
    matchPrefixes: ['partners'],
    children: [
      { id: 'users', label: 'Utilisateurs', path: 'partners/users' },
      { id: 'demo-requests', label: 'Demandes de démonstration', path: 'partners/demo-requests' },
    ],
  },
  {
    id: 'finance',
    label: 'Finance',
    path: 'finance',
    matchPrefixes: ['finance'],
    children: [
      { id: 'finance-overview', label: 'Vue d’ensemble', path: 'finance', end: true },
      { id: 'releves', label: 'Relevés', path: 'finance/releves' },
      { id: 'config', label: 'Entreprises', path: 'finance/config' },
    ],
  },
  {
    id: 'configuration',
    label: 'Configuration',
    path: 'configuration',
    matchPrefixes: ['configuration'],
    children: [],
  },
  {
    id: 'advanced',
    label: 'Outils avancés',
    path: 'advanced',
    matchPrefixes: ['advanced'],
    children: [
      {
        id: 'platform-overview',
        label: 'Vue globale',
        path: 'advanced/platform/overview',
        platformCapability: 'overview',
      },
      {
        id: 'platform-tenants',
        label: 'Tenants',
        path: 'advanced/platform/tenants',
        platformCapability: 'tenants',
      },
      {
        id: 'platform-runbooks',
        label: 'Runbooks',
        path: 'advanced/platform/runbooks',
        platformCapability: 'runbooks',
      },
      {
        id: 'platform-audit',
        label: 'Audit et replay',
        path: 'advanced/platform/audit',
        platformCapability: 'audit',
      },
      {
        id: 'platform-runtime',
        label: 'Runtime',
        path: 'advanced/platform/runtime',
        platformCapability: 'runtime',
      },
      {
        id: 'platform-reconciliation',
        label: 'Réconciliation',
        path: 'advanced/platform/reconciliation',
        platformCapability: 'reconciliation',
      },
      {
        id: 'platform-investigation',
        label: 'Investigation',
        path: 'advanced/platform/investigation',
        platformCapability: 'investigation',
      },
      { id: 'labs-shadow', label: 'Shadow Mode MDI', path: 'advanced/labs/shadow-mode', adminCapability: 'admin.labs.read' },
      { id: 'labs-optuna', label: 'Optimisation Optuna', path: 'advanced/labs/optuna', adminCapability: 'admin.labs.read' },
    ],
  },
];

/**
 * Extrait le chemin relatif sous /dashboard/admin/:publicId
 * @param {string} pathname
 * @param {string} publicId
 */
export function getAdminRelativePath(pathname, publicId) {
  const base = `/dashboard/admin/${publicId}`;
  if (!pathname.startsWith(base)) return '';
  const rest = pathname.slice(base.length).replace(/^\//, '');
  return rest;
}

/**
 * Résout le workspace actif à partir du chemin relatif admin.
 * @param {string} relativePath
 * @returns {AdminWorkspace}
 */
export function resolveActiveWorkspace(relativePath) {
  const rel = (relativePath || '').replace(/\/+$/, '');
  if (!rel) {
    return ADMIN_WORKSPACES.find((w) => w.id === 'overview');
  }
  const first = rel.split('/')[0];
  const found = ADMIN_WORKSPACES.find(
    (w) => w.id !== 'overview' && w.matchPrefixes.some((p) => p === first)
  );
  return found || ADMIN_WORKSPACES.find((w) => w.id === 'overview');
}

/**
 * Indique si la sidebar doit afficher « Outils avancés » (au moins un segment platform accessible
 * ou toujours visible pour labs — les labs restent visibles ; platform filtrée dans workspace nav).
 * @param {(segment: string) => boolean} canAccess
 * @param {boolean} platformLoading
 */
export function shouldShowAdvancedWorkspace(canAccess, platformLoading) {
  if (platformLoading) return true;
  return true;
}
