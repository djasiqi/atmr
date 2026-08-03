import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Link, NavLink, useLocation, useParams } from 'react-router-dom';
import {
  FaHome,
  FaUser,
  FaCar,
  FaFileInvoice,
  FaCog,
  FaTools,
  FaSignOutAlt,
  FaChevronDown,
} from 'react-icons/fa';
import { usePlatformCapabilities } from '../../../../hooks/usePlatformCapabilities';
import { logoutUser } from '../../../../utils/apiClient';
import { getActiveUser } from '../../../../utils/webAuthSession';
import {
  ADMIN_WORKSPACES,
  getAdminRelativePath,
  resolveActiveWorkspace,
} from '../../../../pages/admin/navigation/adminNavRegistry';
import { adminPaths, adminBasePath } from '../../../../pages/admin/routing/adminRoutePaths';
import styles from './AdminSidebar.module.css';

function getInitials(name = '') {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || '').join('') || 'AD';
}

const WORKSPACE_ICONS = {
  overview: FaHome,
  operations: FaCar,
  partners: FaUser,
  finance: FaFileInvoice,
  configuration: FaCog,
  advanced: FaTools,
};

/**
 * Cible sidebar pour un workspace (première page utile).
 * @param {string} publicId
 * @param {import('../../../../pages/admin/navigation/adminNavRegistry').AdminWorkspace} workspace
 * @param {(s: string) => boolean} canAccess
 */
function workspaceHref(publicId, workspace, canAccess) {
  if (workspace.id === 'overview') return adminPaths.overview(publicId);
  if (workspace.id === 'operations') return adminPaths.operationsBookings(publicId);
  if (workspace.id === 'partners') return adminPaths.partnersUsers(publicId);
  if (workspace.id === 'finance') return adminPaths.finance(publicId);
  if (workspace.id === 'configuration') return adminPaths.configuration(publicId);
  if (workspace.id === 'advanced') {
    const firstPlatform = (workspace.children || []).find(
      (c) => c.platformCapability && canAccess(c.platformCapability)
    );
    if (firstPlatform) {
      return `${adminBasePath(publicId)}/${firstPlatform.path}`;
    }
    return adminPaths.advancedLabsShadowMode(publicId);
  }
  return `${adminBasePath(publicId)}/${workspace.path}`;
}

const AdminSidebar = () => {
  const { public_id } = useParams();
  const location = useLocation();
  const adminId = public_id ?? '';
  const { canAccess } = usePlatformCapabilities();

  const showAdvanced = true;

  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target)) {
        setUserMenuOpen(false);
      }
    };
    if (userMenuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [userMenuOpen]);

  const [userData, setUserData] = useState(() => {
    try {
      return getActiveUser();
    } catch {
      return null;
    }
  });

  useEffect(() => {
    const syncUser = () => {
      try {
        setUserData(getActiveUser());
      } catch {
        setUserData(null);
      }
    };
    syncUser();
    window.addEventListener('auth-changed', syncUser);
    return () => window.removeEventListener('auth-changed', syncUser);
  }, []);

  const displayName = userData?.username || 'Administrateur';
  const userEmail = userData?.email || '';
  const initials = useMemo(() => getInitials(displayName), [displayName]);

  const handleLogout = useCallback(async () => {
    await logoutUser();
  }, []);

  const relative = getAdminRelativePath(location.pathname, adminId);
  const activeWorkspace = resolveActiveWorkspace(relative);
  const overviewHref = adminPaths.overview(adminId);

  const workspaces = useMemo(
    () => ADMIN_WORKSPACES.filter((w) => (w.id === 'advanced' ? showAdvanced : true)),
    [showAdvanced]
  );

  return (
    <aside className={styles.sidebar} aria-label="Navigation administration">
      <div className={styles.sidebarBrand}>
        <Link to={overviewHref} className={styles.brandLink} title="Tableau de bord admin">
          <img src="/icon-dark.png" alt="Lirie" className={styles.brandLogo} />
          <div className={styles.brandText}>
            <span className={styles.brandName}>Lirie</span>
            <span className={styles.brandSub}>Administration</span>
          </div>
        </Link>
      </div>

      <nav className={styles.nav} aria-label="Espaces de travail">
        {workspaces.map((workspace) => {
          const Icon = WORKSPACE_ICONS[workspace.id] || FaHome;
          const to = workspaceHref(adminId, workspace, canAccess);
          const isActive = activeWorkspace?.id === workspace.id;
          return (
            <NavLink
              key={workspace.id}
              to={to}
              end={workspace.id === 'overview'}
              className={() => `${styles.navItem} ${isActive ? styles.navActive : ''}`}
              title={workspace.label}
            >
              <span className={styles.navIcon}>
                <Icon />
              </span>
              <span className={styles.navText}>{workspace.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className={styles.sidebarSpacer} aria-hidden="true" />

      <div className={styles.userBlock} ref={userMenuRef}>
        <button
          type="button"
          className={styles.userBtn}
          onClick={() => setUserMenuOpen((p) => !p)}
          aria-expanded={userMenuOpen}
          aria-label="Menu utilisateur"
        >
          <div className={styles.userAvatar}>{initials}</div>
          <div className={styles.userMeta}>
            <span className={styles.userDisplayName}>{displayName}</span>
            <span className={styles.userRole}>Administrateur</span>
          </div>
          <FaChevronDown
            className={`${styles.userChevron} ${userMenuOpen ? styles.userChevronOpen : ''}`}
          />
        </button>

        {userMenuOpen && (
          <div className={styles.userDropdown}>
            {userEmail ? <div className={styles.userDropdownEmail}>{userEmail}</div> : null}
            {userEmail ? <div className={styles.userDropdownDivider} /> : null}
            <button type="button" className={styles.userDropdownItem} onClick={handleLogout}>
              <FaSignOutAlt />
              <span>Déconnexion</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};

export default AdminSidebar;
