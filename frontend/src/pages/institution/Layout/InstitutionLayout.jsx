// pages/institution/Layout/InstitutionLayout.jsx
/**
 * Layout commun pour toutes les pages du portail Institution.
 *
 * Structure :
 * - Sidebar: branding, navigation principale, section secondaire, bloc utilisateur
 * - Header: breadcrumb contextuel, indicateur activité, notifications, CTA
 * - Content: <Outlet />
 */

import React, { useState, useRef, useEffect, useCallback, useMemo, lazy, Suspense } from 'react';
import { useParams, NavLink, Link, Outlet, useNavigate, useLocation } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import {
  FaSignOutAlt,
  FaChevronDown,
  FaChevronRight,
} from 'react-icons/fa';
import {
  HiOutlineViewGrid,
  HiOutlineDocumentText,
  HiOutlineUserGroup,
  HiOutlineCog,
  HiOutlinePlus,
  HiOutlineMenu,
} from 'react-icons/hi';
import { useInstitutionMe, useInstitutionRequests } from '../../../hooks/useInstitutionData';
import useInstitutionSocket from '../../../hooks/useInstitutionSocket';
import {
  getRoleLabel,
  canViewSettings,
  canManageRequests,
} from '../../../utils/institutionPermissions';
import { logoutUser } from '../../../services/authService';
import NotificationBell from './NotificationBell';
import Modal from '../../../components/common/Modal';
import { getActiveAccessToken, getAuthEnv } from '../../../utils/webAuthSession';
import styles from './InstitutionLayout.module.css';

const InstitutionRequestCreate = lazy(() => import('../Requests/InstitutionRequestCreate'));

// ── Route → page meta mapping ──────────────────────────────
const PAGE_META = {
  '': { title: 'Tableau de bord', breadcrumb: null },
  requests: { title: 'Transports', breadcrumb: 'Transports' },
  'requests/new': { title: 'Nouveau transport', breadcrumb: 'Nouveau transport', parent: 'requests' },
  patients: { title: 'Gestion des patients', breadcrumb: 'Patients' },
  settings: { title: 'Paramètres', breadcrumb: 'Paramètres' },
};

/** Resolve current page meta from pathname */
function resolvePageMeta(pathname, publicId) {
  const isDemoPath = String(pathname || '').startsWith('/demo/');
  const dashboardRoot = isDemoPath ? '/demo/dashboard' : '/dashboard';
  const base = `${dashboardRoot}/institution/${publicId}`;
  let relative = pathname.replace(base, '').replace(/^\//, '').replace(/\/$/, '');

  // Detail pages: requests/:id → treat as "requests" parent context
  if (/^requests\/[^/]+$/.test(relative) && relative !== 'requests/new') {
    return {
      title: 'Détail du transport',
      breadcrumb: 'Détail',
      parent: 'requests',
      isDetail: true,
    };
  }

  return PAGE_META[relative] || PAGE_META[''];
}

function getCurrentAccessToken() {
  return getActiveAccessToken({ allowLegacy: true });
}

function decodeTokenClaims() {
  const token = getCurrentAccessToken();
  if (!token) return {};
  try {
    return jwtDecode(token) || {};
  } catch {
    return {};
  }
}

function toDemoEmail(rawEmail) {
  const value = String(rawEmail || '').trim().toLowerCase();
  if (!value) return '';
  if (!value.includes('@')) return `demo-${value}@demo.local`;
  const [localPart, domainPart] = value.split('@', 2);
  if (!localPart) return `demo-user@${domainPart || 'demo.local'}`;
  if (localPart.startsWith('demo-')) return `${localPart}@${domainPart || 'demo.local'}`;
  return `demo-${localPart}@${domainPart || 'demo.local'}`;
}

const InstitutionLayout = () => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [showNewRequest, setShowNewRequest] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const userMenuRef = useRef(null);

  const { data: institutionData, isLoading: loadingMe } = useInstitutionMe();
  const { data: requestsData } = useInstitutionRequests({ per_page: 50 });

  // Socket.IO pour les mises à jour temps réel
  useInstitutionSocket(institutionData?.institution?.id);

  const institution = institutionData?.institution;
  const user = institutionData?.user;
  const tokenClaims = decodeTokenClaims();
  let isDemoEnv = false;
  try {
    isDemoEnv = getAuthEnv() === 'demo';
  } catch {
    isDemoEnv = false;
  }

  const rawInstitutionRole =
    institutionData?.institution_role ||
    tokenClaims?.institution_role ||
    user?.institution_role ||
    null;
  const institutionRole = rawInstitutionRole || (isDemoEnv ? 'institution_admin' : null);

  const resolvedEmail = useMemo(() => {
    const candidate =
      user?.email ||
      tokenClaims?.email ||
      institution?.contact_email ||
      '';
    if (!candidate) return '';
    return isDemoEnv ? toDemoEmail(candidate) : String(candidate).trim();
  }, [user?.email, tokenClaims?.email, institution?.contact_email, isDemoEnv]);

  // Liste des demandes (stabilisée pour les hooks dépendants)
  const allRequests = useMemo(
    () => requestsData?.requests || requestsData?.items || [],
    [requestsData]
  );

  // Badge count: demandes en attente (SENT)
  const pendingCount = useMemo(
    () => allRequests.filter((r) => r.status === 'SENT').length,
    [allRequests]
  );

  // Transports du jour
  const todayCount = useMemo(() => {
    const today = new Date().toISOString().slice(0, 10);
    return allRequests.filter((r) => {
      const d = r.transport_date || r.date;
      return d && d.slice(0, 10) === today;
    }).length;
  }, [allRequests]);

  // Page meta
  const pageMeta = useMemo(
    () => resolvePageMeta(location.pathname, public_id),
    [location.pathname, public_id]
  );
  const isDemoPath = location.pathname.startsWith('/demo/');
  const dashboardRoot = isDemoPath ? '/demo/dashboard' : '/dashboard';
  const basePath = `${dashboardRoot}/institution/${public_id}`;

  useEffect(() => { setMobileOpen(false); }, [location.pathname]);

  // Fermer le menu utilisateur au clic dehors
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target)) {
        setUserMenuOpen(false);
      }
    };
    if (userMenuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [userMenuOpen]);

  const handleLogout = useCallback(async () => {
    try {
      await logoutUser({ redirect: true });
    } catch {
      // Forcer un rechargement complet pour purger tout état React en mémoire
      window.location.href = '/login';
    }
  }, []);

  // User initials for avatar
  const initials = (() => {
    const f = user?.first_name?.[0] || '';
    const l = user?.last_name?.[0] || '';
    if (f || l) return `${f}${l}`.toUpperCase();
    if (resolvedEmail) return (resolvedEmail[0] || '?').toUpperCase();
    return '?';
  })();

  const displayName = (() => {
    if (user?.first_name && user?.last_name) return `${user.first_name} ${user.last_name}`;
    if (user?.first_name) return user.first_name;
    if (isDemoEnv && resolvedEmail) return 'Compte démonstration';
    return resolvedEmail || '';
  })();

  // Navigation principale
  const mainNav = [
    {
      path: `${basePath}`,
      label: 'Tableau de bord',
      icon: <HiOutlineViewGrid />,
      end: true,
    },
    {
      path: `${basePath}/requests`,
      label: 'Transports',
      icon: <HiOutlineDocumentText />,
      badge: pendingCount > 0 ? pendingCount : null,
    },
    {
      path: `${basePath}/patients`,
      label: 'Patients',
      icon: <HiOutlineUserGroup />,
    },
  ];

  // Navigation secondaire
  const secondaryNav = [];
  if (canViewSettings(institutionRole)) {
    secondaryNav.push({
      path: `${basePath}/settings`,
      label: 'Paramètres',
      icon: <HiOutlineCog />,
    });
  }

  // Breadcrumb items
  const breadcrumbs = useMemo(() => {
    const items = [{ label: institution?.name || 'Institution', path: basePath }];
    if (pageMeta.parent) {
      const parentMeta = PAGE_META[pageMeta.parent];
      if (parentMeta) {
        items.push({ label: parentMeta.breadcrumb, path: `${basePath}/${pageMeta.parent}` });
      }
    }
    if (pageMeta.breadcrumb) {
      items.push({ label: pageMeta.breadcrumb, path: null });
    }
    return items;
  }, [institution?.name, basePath, pageMeta]);

  return (
    <div className={styles.container}>
      {/* ── Sidebar ───────────────────────────────────────── */}
      <aside className={`${styles.sidebar} ${mobileOpen ? styles.sidebarOpen : ''}`} data-tour-id="institution-sidebar">
        {/* Branding */}
        <div className={styles.sidebarBrand}>
          <img src="/icon-dark.png" alt="Lirie" className={styles.brandLogo} />
          <div className={styles.brandText}>
            <span className={styles.brandName}>Lirie</span>
            <span className={styles.brandSub}>Portail Institution</span>
          </div>
        </div>

        {/* Navigation principale */}
        <div className={styles.navSection}>
          <div className={styles.navLabel}>Navigation</div>
          <nav className={styles.nav}>
            {mainNav.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.end || false}
                className={({ isActive }) =>
                  `${styles.navItem} ${isActive ? styles.navActive : ''}`
                }
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span className={styles.navText}>{item.label}</span>
                {item.badge && (
                  <span className={styles.navBadge}>{item.badge}</span>
                )}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Navigation secondaire */}
        {secondaryNav.length > 0 && (
          <div className={styles.navSection}>
            <div className={styles.navDivider} />
            <nav className={styles.nav}>
              {secondaryNav.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `${styles.navItem} ${isActive ? styles.navActive : ''}`
                  }
                >
                  <span className={styles.navIcon}>{item.icon}</span>
                  <span className={styles.navText}>{item.label}</span>
                </NavLink>
              ))}
            </nav>
          </div>
        )}

        {/* Spacer */}
        <div className={styles.sidebarSpacer} />

        {/* Bloc utilisateur */}
        <div className={styles.userBlock} ref={userMenuRef}>
          <button
            className={styles.userBtn}
            onClick={() => setUserMenuOpen((p) => !p)}
          >
            <div className={styles.userAvatar}>{initials}</div>
            <div className={styles.userMeta}>
              <span className={styles.userDisplayName}>{displayName}</span>
              <span className={styles.userRole}>
                {getRoleLabel(institutionRole)}
              </span>
            </div>
            <FaChevronDown className={`${styles.userChevron} ${userMenuOpen ? styles.userChevronOpen : ''}`} />
          </button>

          {userMenuOpen && (
            <div className={styles.userDropdown}>
              <div className={styles.userDropdownEmail}>{resolvedEmail || user?.email || ''}</div>
              <div className={styles.userDropdownDivider} />
              <button className={styles.userDropdownItem} onClick={handleLogout}>
                <FaSignOutAlt />
                <span>Se déconnecter</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* ── Backdrop mobile ── */}
      {mobileOpen && (
        <div className={styles.backdrop} onClick={() => setMobileOpen(false)} />
      )}

      {/* ── Main content area ─────────────────────────────── */}
      <main className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          {/* Zone gauche — Contexte */}
          <div className={styles.headerLeft}>
            <button
              className={styles.hamburgerBtn}
              onClick={() => setMobileOpen((p) => !p)}
              aria-label="Menu"
            >
              <HiOutlineMenu />
            </button>
            <div className={styles.headerContext}>
              {/* Breadcrumb */}
              {breadcrumbs.length > 1 && (
                <nav className={styles.breadcrumb} aria-label="Fil d'Ariane">
                  {breadcrumbs.map((item, i) => (
                    <React.Fragment key={i}>
                      {i > 0 && <FaChevronRight className={styles.breadcrumbSep} />}
                      {item.path && i < breadcrumbs.length - 1 ? (
                        <Link to={item.path} className={styles.breadcrumbLink}>
                          {item.label}
                        </Link>
                      ) : (
                        <span className={styles.breadcrumbCurrent}>{item.label}</span>
                      )}
                    </React.Fragment>
                  ))}
                </nav>
              )}
              {/* Titre page */}
              <h1 className={styles.pageTitle}>
                {loadingMe ? '' : pageMeta.title}
              </h1>
            </div>
          </div>

          {/* Zone droite — Actions */}
          <div className={styles.headerRight}>
            {/* Indicateur activité du jour */}
            {todayCount > 0 && (
              <div className={styles.headerIndicator}>
                <span className={styles.indicatorDot} />
                <span className={styles.indicatorText}>
                  {todayCount} transport{todayCount > 1 ? 's' : ''} aujourd'hui
                </span>
              </div>
            )}
            {pendingCount > 0 && (
              <div className={`${styles.headerIndicator} ${styles.indicatorPending}`}>
                <span className={styles.indicatorDot} />
                <span className={styles.indicatorText}>
                  {pendingCount} en attente
                </span>
              </div>
            )}

            <div className={styles.headerDivider} />

            <NotificationBell />

            {canManageRequests(institutionRole) && (
              <button
                type="button"
                className={styles.newRequestBtn}
                onClick={() => setShowNewRequest(true)}
                data-tour-id="institution-create-request-cta"
              >
                <HiOutlinePlus className={styles.newRequestIcon} />
                <span>Nouvelle demande</span>
              </button>
            )}
          </div>
        </header>

        {/* Child route content */}
        <div className={styles.content}>
          <Outlet />
        </div>

        {/* Modal nouvelle demande — accessible depuis toutes les pages */}
        {showNewRequest && (
          <Modal onClose={() => setShowNewRequest(false)} size="xl">
            <Suspense fallback={<div style={{ padding: 40, textAlign: 'center', color: '#94A3B8' }}>Chargement...</div>}>
              <InstitutionRequestCreate
                onClose={() => setShowNewRequest(false)}
                onSuccess={() => {
                  setShowNewRequest(false);
                  navigate(`${basePath}/requests`);
                }}
              />
            </Suspense>
          </Modal>
        )}
      </main>
    </div>
  );
};

export default InstitutionLayout;
