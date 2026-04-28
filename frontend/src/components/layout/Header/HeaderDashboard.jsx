// src/components/layout/Header/HeaderDashboard.jsx
import React, { useState, useEffect, useRef, useMemo, useId, useCallback } from 'react';
import { Link, NavLink, useNavigate, useParams } from 'react-router-dom';
import {
  FiBell,
  FiCalendar,
  FiChevronDown,
  FiGrid,
  FiHelpCircle,
  FiList,
  FiLogOut,
  FiSettings,
  FiUser,
} from 'react-icons/fi';
import { toast } from 'sonner';
import styles from './HeaderDashboard.module.css';
import { logoutUser } from '../../../utils/apiClient';
import { getEnvUser } from '../../../utils/webAuthSession';

function formatToday() {
  return new Date().toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
}

function initialsFromName(name) {
  if (!name || !String(name).trim()) return '?';
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
  }
  return parts[0].slice(0, 2).toUpperCase();
}

/**
 * @param {{ variant?: 'default' | 'admin' }} props
 * - default : barre pleine largeur (client, chauffeur).
 * - admin : alignée CompanyHeader — uniquement au-dessus du contenu, pas de la sidebar teal.
 */
const HeaderDashboard = ({ variant = 'default', userName: userNameProp }) => {
  const isAdmin = variant === 'admin';
  const { public_id } = useParams();

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [userName, setUserName] = useState(userNameProp || 'Utilisateur');
  const [publicId, setPublicId] = useState(null);
  const [userRole, setUserRole] = useState('');
  const navigate = useNavigate();
  const menuRef = useRef(null);
  const menuPanelId = useId();
  const closeMenu = useCallback(() => setIsMenuOpen(false), []);

  useEffect(() => {
    const user = getEnvUser();
    if (user) {
      try {
        if (user?.username) setUserName(user.username);
        if (user?.public_id) setPublicId(user.public_id);
        if (user?.role) setUserRole(user.role);
      } catch (error) {
        console.error("Erreur lors de la récupération de l'utilisateur :", error);
      }
    }
  }, []);

  useEffect(() => {
    if (userNameProp) setUserName(userNameProp);
  }, [userNameProp]);

  const displayName = userNameProp || userName;

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleAccountClick = () => {
    setIsMenuOpen(false);
    if (!publicId) return;
    const r = (userRole || '').toLowerCase();
    if (r === 'driver') {
      navigate('/driver/settings');
      return;
    }
    navigate(`/dashboard/account/${publicId}`);
  };

  const handleLogout = async () => {
    setIsMenuOpen(false);
    await logoutUser();
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isMenuOpen) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') setIsMenuOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [isMenuOpen]);

  const roleLower = (userRole || '').toLowerCase();
  const dashboardLink =
    publicId && roleLower
      ? `/dashboard/${roleLower}/${publicId}`
      : '/dashboard';

  const clientNavItems =
    publicId && roleLower === 'client'
      ? [
          { to: `/dashboard/client/${publicId}`, label: 'Réserver' },
          { to: `/reservations/${publicId}`, label: 'Mes courses' },
          { to: `/dashboard/account/${publicId}`, label: 'Mon compte' },
          { to: '/aide', label: 'Aide' },
        ]
      : [];

  const driverNavItems =
    publicId && roleLower === 'driver'
      ? [
          { to: `/dashboard/driver/${publicId}`, label: 'Tableau de bord' },
          { to: '/driver/schedule', label: 'Planning' },
          { to: '/driver/map', label: 'Carte' },
          { to: '/driver/history', label: 'Historique' },
          { to: '/driver/settings', label: 'Paramètres' },
        ]
      : [];

  const quickNavItems =
    roleLower === 'client' ? clientNavItems : roleLower === 'driver' ? driverNavItems : [];
  const adminHome =
    public_id && isAdmin ? `/dashboard/admin/${public_id}` : dashboardLink;

  const todayLabel = useMemo(() => formatToday(), []);

  const roleLabel =
    roleLower === 'driver' ? 'Chauffeur' : roleLower === 'client' ? 'Client' : 'Compte';

  const handleNotificationsClick = () => {
    toast.info('Aucune notification pour le moment.');
  };

  const userDropdown = (
    <div className={styles.userSection} ref={menuRef}>
      <button
        type="button"
        className={`${styles.userButton}${isMenuOpen ? ` ${styles.userButtonOpen}` : ''}`}
        onClick={toggleMenu}
        aria-expanded={isMenuOpen}
        aria-haspopup="true"
        aria-controls={menuPanelId}
        aria-label={`Menu compte — ${displayName}`}
      >
        <span className={styles.userButtonAvatar} aria-hidden>
          {initialsFromName(displayName)}
        </span>
        <span className={styles.userButtonName}>{displayName}</span>
        <FiChevronDown className={styles.userButtonChevron} aria-hidden />
      </button>
      {isMenuOpen && (
        <div
          id={menuPanelId}
          className={styles.dropdownMenu}
          role="region"
          aria-label="Actions du compte"
        >
          <div className={styles.userInfo}>
            <div className={styles.userAvatar} aria-hidden>
              {initialsFromName(displayName)}
            </div>
            <div className={styles.userMeta}>
              <p className={styles.userName}>{displayName}</p>
              <p className={styles.userRole}>{roleLabel}</p>
            </div>
          </div>
          <ul className={styles.menuList}>
            <li>
              <button type="button" className={styles.menuItem} onClick={handleAccountClick}>
                {roleLower === 'driver' ? (
                  <FiSettings className={styles.menuItemIcon} aria-hidden />
                ) : (
                  <FiUser className={styles.menuItemIcon} aria-hidden />
                )}
                <span>{roleLower === 'driver' ? 'Paramètres' : 'Gestion du compte'}</span>
              </button>
            </li>
            {publicId && roleLower === 'client' ? (
              <li>
                <Link
                  to={`/reservations/${publicId}`}
                  className={styles.menuItem}
                  onClick={closeMenu}
                >
                  <FiList className={styles.menuItemIcon} aria-hidden />
                  <span>Mes courses</span>
                </Link>
              </li>
            ) : null}
            {publicId && roleLower === 'client' ? (
              <li>
                <Link
                  to={`/dashboard/client/${publicId}`}
                  className={styles.menuItem}
                  onClick={closeMenu}
                >
                  <FiCalendar className={styles.menuItemIcon} aria-hidden />
                  <span>Réserver</span>
                </Link>
              </li>
            ) : null}
            {publicId && roleLower === 'driver' ? (
              <li>
                <Link
                  to={`/dashboard/driver/${publicId}`}
                  className={styles.menuItem}
                  onClick={closeMenu}
                >
                  <FiGrid className={styles.menuItemIcon} aria-hidden />
                  <span>Tableau de bord</span>
                </Link>
              </li>
            ) : null}
            <li>
              <Link to="/aide" className={styles.menuItem} onClick={closeMenu}>
                <FiHelpCircle className={styles.menuItemIcon} aria-hidden />
                <span>Aide &amp; support</span>
              </Link>
            </li>
          </ul>
          <div className={styles.logout}>
            <button type="button" className={styles.logoutButton} onClick={handleLogout}>
              <FiLogOut className={styles.menuItemIcon} aria-hidden />
              <span>Déconnexion</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );

  if (isAdmin) {
    return (
      <header
        className={`${styles.header} ${styles.headerAdmin}`}
        role="banner"
        aria-label="En-tête console administrateur"
      >
        <div className={styles.headerLeft}>
          <Link to={adminHome} className={styles.brand} aria-label="Accueil console administrateur">
            <div className={styles.logoWrap}>
              <img src="/icon-dark.png" alt="" className={styles.logoImg} />
            </div>
            <span className={styles.adminWorkspaceTitle}>Console administrateur</span>
          </Link>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.indicator}>
            <FiCalendar className={styles.indicatorIcon} aria-hidden />
            <span className={styles.indicatorText}>{todayLabel}</span>
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className={styles.header} role="banner">
      <Link to={dashboardLink} className={styles.logo} aria-label="Aller au tableau de bord">
        <img src="/logo-lirie.png" alt="" className={styles.logoBrandImg} width="120" height="30" />
      </Link>
      <nav className={styles.nav} aria-label="Navigation rapide">
        <ul className={styles.navList}>
          {quickNavItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                className={({ isActive }) => `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`}
                end
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <div className={styles.headerActions}>
        <button
          type="button"
          className={styles.bellButton}
          onClick={handleNotificationsClick}
          aria-label="Notifications"
        >
          <FiBell className={styles.bellIcon} aria-hidden />
        </button>
        {userDropdown}
      </div>
    </header>
  );
};

export default HeaderDashboard;
