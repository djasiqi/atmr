// src/components/layout/Sidebar/AdminSidebar/AdminSidebar.js
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Link, NavLink, useLocation, useParams, useNavigate } from 'react-router-dom';
import {
  FaHome,
  FaUser,
  FaCar,
  FaFileInvoice,
  FaCog,
  FaRobot,
  FaChartLine,
  FaServer,
  FaSignOutAlt,
  FaChevronDown,
} from 'react-icons/fa';
import { usePlatformCapabilities, PLATFORM_SEGMENTS } from '../../../../hooks/usePlatformCapabilities';
import { logoutUser } from '../../../../utils/apiClient';
import { getActiveUser } from '../../../../utils/webAuthSession';
import styles from './AdminSidebar.module.css';

function getInitials(name = '') {
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() || '').join('') || 'AD';
}

/** Largeurs alignées sur CompanySidebar + --sidebar-w pour le contenu principal */
function getAdminSidebarWidthPx() {
  const w = window.innerWidth;
  if (w <= 480) return 56;
  if (w <= 768) return 64;
  if (w <= 1024) return 220;
  return 256;
}

const AdminSidebar = () => {
  const { public_id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const adminId = public_id ?? '';
  const { canAccess, isLoading: platformLoading } = usePlatformCapabilities();

  const showPlatformNav = useMemo(() => {
    if (platformLoading) return true;
    return PLATFORM_SEGMENTS.some((s) => canAccess(s));
  }, [platformLoading, canAccess]);

  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef(null);

  useEffect(() => {
    const update = () => {
      document.documentElement.style.setProperty('--sidebar-w', `${getAdminSidebarWidthPx()}px`);
    };
    update();
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('resize', update);
      document.documentElement.style.setProperty('--sidebar-w', '72px');
    };
  }, []);

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

  const publicId = userData?.public_id || null;
  const displayName = userData?.username || 'Administrateur';
  const userEmail = userData?.email || '';
  const initials = useMemo(() => getInitials(displayName), [displayName]);

  const handleLogout = useCallback(async () => {
    await logoutUser();
  }, []);

  const handleAccountClick = () => {
    if (!publicId) return;
    navigate(`/dashboard/account/${publicId}`);
    setUserMenuOpen(false);
  };

  const base = `/dashboard/admin/${adminId}`;

  const sections = [
    {
      title: 'Exploitation métier',
      items: [
        { icon: FaHome, label: 'Tableau de bord', to: base, end: true, isPlatform: false },
        { icon: FaCar, label: 'Réservations', to: `${base}/reservations`, isPlatform: false },
        {
          icon: FaFileInvoice,
          label: 'Facturation',
          to: `${base}/billing/pilotage`,
          isBillingHub: true,
          isPlatform: false,
        },
        { icon: FaChartLine, label: 'Demandes demo', to: `${base}/demo-requests`, isPlatform: false },
      ],
    },
    {
      title: 'Administration applicative',
      items: [
        { icon: FaUser, label: 'Utilisateurs', to: `${base}/users`, isPlatform: false },
        { icon: FaCog, label: 'Paramètres', to: `${base}/settings`, isPlatform: false },
      ],
    },
    {
      title: 'Intelligence / optimisation',
      items: [
        { icon: FaRobot, label: 'Shadow Mode MDI', to: `${base}/shadow-mode`, isPlatform: false },
        { icon: FaChartLine, label: 'Optimisation Optuna', to: `${base}/optuna`, isPlatform: false },
      ],
    },
  ];

  if (showPlatformNav) {
    sections.push({
      title: 'Plateforme',
      items: [
        {
          icon: FaServer,
          label: 'Ops / Platform',
          to: `${base}/platform-ops/overview`,
          end: false,
          isPlatform: true,
        },
      ],
    });
  }

  return (
    <aside className={styles.sidebar} aria-label="Navigation administration">
      <div className={styles.sidebarBrand}>
        <Link to={base} className={styles.brandLink} title="Tableau de bord admin">
          <img src="/icon-dark.png" alt="Lirie" className={styles.brandLogo} />
          <div className={styles.brandText}>
            <span className={styles.brandName}>Lirie</span>
            <span className={styles.brandSub}>Administration</span>
          </div>
        </Link>
      </div>

      {sections.map((section, idx) => (
        <div key={section.title} className={styles.navSection}>
          {idx > 0 && <div className={styles.navDivider} aria-hidden="true" />}
          <div className={styles.navLabel}>{section.title}</div>
          <nav className={styles.nav} aria-label={section.title}>
            {section.items.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end ?? false}
                  className={({ isActive }) => {
                    const active = item.isPlatform
                      ? location.pathname.includes('/platform-ops')
                      : item.isBillingHub
                        ? location.pathname.startsWith(`${base}/billing`)
                        : isActive;
                    return `${styles.navItem} ${active ? styles.navActive : ''}`;
                  }}
                  title={item.label}
                >
                  <span className={styles.navIcon}>
                    <Icon />
                  </span>
                  <span className={styles.navText}>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      ))}

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
            <button type="button" className={styles.userDropdownItem} onClick={handleAccountClick}>
              Gestion du compte
            </button>
            {publicId ? (
              <Link
                to={`/reservations/${publicId}`}
                className={styles.userDropdownItem}
                onClick={() => setUserMenuOpen(false)}
              >
                Mes réservations
              </Link>
            ) : null}
            <Link
              to="/dashboard/support"
              className={styles.userDropdownItem}
              onClick={() => setUserMenuOpen(false)}
            >
              Support client
            </Link>
            <Link
              to="/dashboard/upcoming-rides"
              className={styles.userDropdownItem}
              onClick={() => setUserMenuOpen(false)}
            >
              Prochaines courses
            </Link>
            <div className={styles.userDropdownDivider} />
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
