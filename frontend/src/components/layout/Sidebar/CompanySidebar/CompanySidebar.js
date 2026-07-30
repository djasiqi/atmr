// src/components/layout/Sidebar/CompanySidebar/CompanySidebar.js
/**
 * Sidebar professionnelle — Espace Entreprise
 *
 * Design aligné sur le portail Institution :
 * - Branding Lirie en haut
 * - Navigation segmentée (principale + outils)
 * - Bloc utilisateur en bas
 * - Fond brand (#00796B)
 * - Responsive automatique : ouvert grand écran, icônes seules petit écran
 */

import React, { useEffect, useLayoutEffect, useRef, useCallback, useMemo, useState } from 'react';
import { NavLink, useParams, useLocation } from 'react-router-dom';
import {
  HiOutlineViewGrid,
  HiOutlineClipboardList,
  HiOutlineUserCircle,
  HiOutlineUserGroup,
  HiOutlineDocumentText,
  HiOutlineLightningBolt,
  HiOutlineChartBar,
  HiOutlineCog,
} from 'react-icons/hi';
import { FaSignOutAlt, FaChevronDown } from 'react-icons/fa';
import { useLirieCompany } from '../../../../hooks/useLirieCompany';
import { logoutUser } from '../../../../utils/apiClient';
import { getActiveUser, getAuthEnv } from '../../../../utils/webAuthSession';
import { prefetchCompanyRouteChunk } from '../../CompanyEnterpriseLayout';
import styles from './CompanySidebar.module.css';

const ROUTE_CHUNK_LOADERS = {
  '': () => import('../../../../pages/company/Dashboard/CompanyDashboard'),
  reservations: () => import('../../../../pages/company/Reservations/CompanyReservations'),
  drivers: () => import('../../../../pages/company/Driver/CompanyDriver'),
  clients: () => import('../../../../pages/company/Clients/CompanyClients'),
  'invoices/clients': () => import('../../../../pages/company/Invoices/ClientInvoices'),
  invoices: () => import('../../../../pages/company/Invoices/CompanyInvoices'),
  dispatch: () => import('../../../../pages/company/Dispatch/UnifiedDispatchRefactored'),
  analytics: () => import('../../../../pages/company/Analytics/AnalyticsDashboard'),
  settings: () => import('../../../../pages/company/Settings/CompanySettings'),
  planning: () => import('../../../../pages/company/Planning/CompanyPlanning'),
};

function prefetchForNavPath(path) {
  if (!path || typeof path !== 'string') return;
  const segments = path.replace(/\/$/, '').split('/').filter(Boolean);
  const idx = segments.indexOf('company');
  const relative = idx >= 0 ? segments.slice(idx + 2).join('/') : '';
  const loader =
    ROUTE_CHUNK_LOADERS[relative] ||
    ROUTE_CHUNK_LOADERS[relative.split('/')[0]] ||
    (relative === '' ? ROUTE_CHUNK_LOADERS[''] : null);
  if (loader) prefetchCompanyRouteChunk(loader);
}
function getInitials(name = '') {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || '').join('') || 'CO';
}

// Breakpoint widths — synchronized with CSS media queries
function getSidebarWidth() {
  const w = window.innerWidth;
  if (w <= 480) return 56;
  if (w <= 768) return 64;
  if (w <= 1024) return 220;
  return 256;
}

const CompanySidebar = () => {
  const params = useParams();
  const location = useLocation();

  const public_id =
    params.public_id ||
    (() => {
      const match = location.pathname.match(/(?:\/demo)?\/dashboard\/company\/([^/]+)/);
      return match?.[1] || null;
    })();

  const { company } = useLirieCompany();
  const companyName = company?.name || 'Entreprise';

  // User dropdown
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);

  // Sync --sidebar-w CSS variable on resize (layout phase to avoid first-paint shift)
  useLayoutEffect(() => {
    const update = () => {
      document.documentElement.style.setProperty('--sidebar-w', `${getSidebarWidth()}px`);
    };
    update();
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('resize', update);
      document.documentElement.style.setProperty('--sidebar-w', '72px');
    };
  }, []);

  // Close user menu on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target)) {
        setUserMenuOpen(false);
      }
    };
    if (userMenuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [userMenuOpen]);

  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = useCallback(async () => {
    if (isLoggingOut) return;
    setIsLoggingOut(true);
    setUserMenuOpen(false);
    try {
      await logoutUser();
    } catch (error) {
      console.error('Échec de la déconnexion:', error);
    } finally {
      setIsLoggingOut(false);
    }
  }, [isLoggingOut]);

  const [userData, setUserData] = useState(() => getActiveUser());

  useEffect(() => {
    const syncUser = () => setUserData(getActiveUser());
    syncUser();
    window.addEventListener('auth-changed', syncUser);
    return () => window.removeEventListener('auth-changed', syncUser);
  }, []);

  const userEmail = userData?.email || '';
  const userRole = userData?.role || 'company';
  const roleLabel = userRole === 'admin' ? 'Administrateur' : 'Entreprise';

  const displayName = useMemo(() => {
    if (userData?.first_name && userData?.last_name)
      return `${userData.first_name} ${userData.last_name}`;
    if (userData?.first_name) return userData.first_name;
    return companyName;
  }, [userData, companyName]);

  const initials = useMemo(() => {
    if (userData?.first_name || userData?.last_name) {
      const f = userData.first_name?.[0] || '';
      const l = userData.last_name?.[0] || '';
      return `${f}${l}`.toUpperCase() || 'CO';
    }
    return getInitials(companyName);
  }, [userData, companyName]);

  if (!public_id) return null;

  const isDemoEnv =
    location.pathname.startsWith('/demo/') ||
    getAuthEnv() === 'demo';
  const dashboardRoot = isDemoEnv ? '/demo/dashboard' : '/dashboard';
  const basePath = `${dashboardRoot}/company/${public_id}`;

  // ── Navigation items ──
  const mainNav = [
    { path: basePath, label: 'Tableau de bord', icon: <HiOutlineViewGrid />, end: true },
    { path: `${basePath}/reservations`, label: 'Réservations', icon: <HiOutlineClipboardList /> },
    { path: `${basePath}/drivers`, label: 'Chauffeurs', icon: <HiOutlineUserCircle /> },
    { path: `${basePath}/clients`, label: 'Gestion Clients', icon: <HiOutlineUserGroup /> },
  ];

  const toolsNav = [
    { path: `${basePath}/invoices/clients`, label: 'Facturation', icon: <HiOutlineDocumentText /> },
    { path: `${basePath}/dispatch`, label: 'Exploitation', icon: <HiOutlineLightningBolt /> },
    { path: `${basePath}/analytics`, label: 'Analytics', icon: <HiOutlineChartBar /> },
  ];

  const secondaryNav = [
    { path: `${basePath}/settings`, label: 'Paramètres', icon: <HiOutlineCog /> },
  ];

  return (
    <aside className={styles.sidebar}>
      {/* ── Branding ── */}
      <div className={styles.sidebarBrand}>
        <img src="/icon-dark.png" alt="Lirie" className={styles.brandLogo} width="36" height="36" />
        <div className={styles.brandText}>
          <span className={styles.brandName}>Lirie</span>
          <span className={styles.brandSub}>Espace Entreprise</span>
        </div>
      </div>

      {/* ── Navigation principale ── */}
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
              aria-label={item.label}
              title={item.label}
              onMouseEnter={() => prefetchForNavPath(item.path)}
              onFocus={() => prefetchForNavPath(item.path)}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              <span className={styles.navText}>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      {/* ── Outils ── */}
      <div className={styles.navSection}>
        <div className={styles.navDivider} />
        <div className={styles.navLabel}>Outils</div>
        <nav className={styles.nav}>
          {toolsNav.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              data-tour-id={
                item.path.endsWith('/dispatch')
                  ? 'sidebar-dispatch-link'
                  : item.label === 'Facturation'
                    ? 'sidebar-facturation-link'
                    : undefined
              }
              className={({ isActive }) =>
                `${styles.navItem} ${isActive ? styles.navActive : ''}`
              }
              aria-label={item.label}
              title={item.label}
              onMouseEnter={() => prefetchForNavPath(item.path)}
              onFocus={() => prefetchForNavPath(item.path)}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              <span className={styles.navText}>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      {/* ── Paramètres ── */}
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
              aria-label={item.label}
              title={item.label}
              onMouseEnter={() => prefetchForNavPath(item.path)}
              onFocus={() => prefetchForNavPath(item.path)}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              <span className={styles.navText}>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      {/* ── Spacer ── */}
      <div className={styles.sidebarSpacer} />

      {/* ── User block ── */}
      <div className={styles.userBlock} ref={userMenuRef}>
        <button
          className={styles.userBtn}
          onClick={() => setUserMenuOpen((p) => !p)}
          aria-label="Menu utilisateur"
          aria-expanded={userMenuOpen}
          aria-haspopup="menu"
          type="button"
        >
          <div className={styles.userAvatar}>{initials}</div>
          <div className={styles.userMeta}>
            <span className={styles.userDisplayName}>{displayName}</span>
            <span className={styles.userRole}>{roleLabel}</span>
          </div>
          <FaChevronDown
            className={`${styles.userChevron} ${userMenuOpen ? styles.userChevronOpen : ''}`}
          />
        </button>

        {userMenuOpen && (
          <div className={styles.userDropdown}>
            {userEmail && <div className={styles.userDropdownEmail}>{userEmail}</div>}
            <div className={styles.userDropdownDivider} />
            <button
              className={styles.userDropdownItem}
              onClick={handleLogout}
              type="button"
              disabled={isLoggingOut}
              aria-busy={isLoggingOut}
            >
              <FaSignOutAlt />
              <span>{isLoggingOut ? 'Déconnexion…' : 'Se déconnecter'}</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};

export default CompanySidebar;
