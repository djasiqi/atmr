// src/components/layout/Header/HeaderDashboard.jsx
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { FiCalendar } from 'react-icons/fi';
import styles from './HeaderDashboard.module.css';
import { logoutUser } from '../../../utils/apiClient';

function formatToday() {
  return new Date().toLocaleDateString('fr-CH', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
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

  useEffect(() => {
    const userData = localStorage.getItem('user');
    if (userData) {
      try {
        const user = JSON.parse(userData);
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
    if (!publicId) return;
    navigate(`/dashboard/account/${publicId}`);
  };

  const handleLogout = async () => {
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

  const dashboardLink = publicId && userRole ? `/dashboard/${userRole}/${publicId}` : '/dashboard';
  const adminHome =
    public_id && isAdmin ? `/dashboard/admin/${public_id}` : dashboardLink;

  const todayLabel = useMemo(() => formatToday(), []);

  const userDropdown = (
    <div className={styles.userSection} ref={menuRef}>
      <button
        type="button"
        className={styles.userButton}
        onClick={toggleMenu}
        aria-expanded={isMenuOpen}
        aria-haspopup="menu"
        aria-label="Menu utilisateur"
      >
        {displayName} <span className={styles.arrow} aria-hidden>
          ▼
        </span>
      </button>
      {isMenuOpen && (
        <div className={styles.dropdownMenu}>
          <div className={styles.userInfo}>
            <p className={styles.userName}>{displayName}</p>
          </div>
          <div className={styles.menuOptions}>
            <button type="button" className={styles.menuLink} onClick={handleAccountClick}>
              Gestion du compte
            </button>
            {publicId ? (
              <Link to={`/reservations/${publicId}`} className={styles.menuLink}>
                Mes Réservations
              </Link>
            ) : (
              <span className={styles.menuLink} style={{ cursor: 'not-allowed', opacity: 0.5 }}>
                Mes Réservations (Indisponible)
              </span>
            )}
            <Link to="/dashboard/support" className={styles.menuLink}>
              Support client
            </Link>
            <Link to="/dashboard/upcoming-rides" className={styles.menuLink}>
              Prochaines courses
            </Link>
          </div>
          <div className={styles.logout}>
            <button type="button" className={styles.logoutButton} onClick={handleLogout}>
              Déconnexion
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
      <Link to={dashboardLink} className={styles.logo}>
        Dashboard
      </Link>
      <nav className={styles.nav} aria-label="Navigation rapide">
        <ul className={styles.navList}>
          <li>
            <Link to="/dashboard/bookings" className={styles.navLink}>
              Mes Réservations
            </Link>
          </li>
          <li>
            <Link to="/dashboard/payments" className={styles.navLink}>
              Paiements
            </Link>
          </li>
          <li>
            <Link to="/dashboard/profile" className={styles.navLink}>
              Profil
            </Link>
          </li>
          <li>
            <Link to="/dashboard/help" className={styles.navLink}>
              Aide
            </Link>
          </li>
        </ul>
      </nav>
      {userDropdown}
    </header>
  );
};

export default HeaderDashboard;
