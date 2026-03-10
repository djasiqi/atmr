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

import React, { useEffect, useRef, useCallback, useMemo, useState } from 'react';
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
import useCompanyData from '../../../../hooks/useCompanyData';
import { logoutUser } from '../../../../utils/apiClient';
import styles from './CompanySidebar.module.css';

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

  const companyData = useCompanyData() || {};
  const company = companyData.company || null;
  const companyName = company?.name || 'Entreprise';

  // User dropdown
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);

  // Sync --sidebar-w CSS variable on resize
  useEffect(() => {
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

  const handleLogout = useCallback(async () => {
    try {
      await logoutUser();
    } catch {
      window.location.href = '/login';
    }
  }, []);

  // User data from localStorage
  const userData = useMemo(() => {
    try {
      const raw = localStorage.getItem('user') || localStorage.getItem('company_user');
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
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
    (localStorage.getItem('lirie_auth_env') || '').toLowerCase() === 'demo';
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
    { path: `${basePath}/dispatch`, label: 'Dispatch', icon: <HiOutlineLightningBolt /> },
    { path: `${basePath}/analytics`, label: 'Analytics', icon: <HiOutlineChartBar /> },
  ];

  const secondaryNav = [
    { path: `${basePath}/settings`, label: 'Paramètres', icon: <HiOutlineCog /> },
  ];

  return (
    <aside className={styles.sidebar}>
      {/* ── Branding ── */}
      <div className={styles.sidebarBrand}>
        <img src="/icon-dark.png" alt="Lirie" className={styles.brandLogo} />
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
                item.label === 'Dispatch'
                  ? 'sidebar-dispatch-link'
                  : item.label === 'Facturation'
                    ? 'sidebar-facturation-link'
                    : undefined
              }
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
            <button className={styles.userDropdownItem} onClick={handleLogout}>
              <FaSignOutAlt />
              <span>Se déconnecter</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};

export default CompanySidebar;
