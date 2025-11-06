// src/components/layout/HeaderDashboard.jsx
import React, { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import styles from "./HeaderDashboard.module.css";
import { logoutUser } from "../../../utils/apiClient";

const HeaderDashboard = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [userName, setUserName] = useState("Utilisateur");
  const [publicId, setPublicId] = useState(null);
  const [userRole, setUserRole] = useState("");
  const navigate = useNavigate();
  const menuRef = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem("authToken");
    const userData = localStorage.getItem("user");
    if (token && userData) {
      try {
        const user = JSON.parse(userData);
        if (user?.username) setUserName(user.username);
        if (user?.public_id) setPublicId(user.public_id);
        if (user?.role) setUserRole(user.role);
      } catch (error) {
        console.error(
          "Erreur lors de la récupération de l'utilisateur :",
          error
        );
      }
    }
  }, []);

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
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const dashboardLink =
    publicId && userRole ? `/dashboard/${userRole}/${publicId}` : "/dashboard";

  return (
    <header className={styles.header}>
      <Link to={dashboardLink} className={styles.logo}>
        Dashboard
      </Link>
      <nav className={styles.nav}>
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
      <div className={styles.userSection} ref={menuRef}>
        <div className={styles.userButton} onClick={toggleMenu}>
          {userName} <span className={styles.arrow}>▼</span>
        </div>
        {isMenuOpen && (
          <div className={styles.dropdownMenu}>
            <div className={styles.userInfo}>
              <p className={styles.userName}>{userName}</p>
            </div>
            <div className={styles.menuOptions}>
              <button className={styles.menuLink} onClick={handleAccountClick}>
                Gestion du compte
              </button>
              {publicId ? (
                <Link
                  to={`/reservations/${publicId}`}
                  className={styles.menuLink}
                >
                  Mes Réservations
                </Link>
              ) : (
                <span
                  className={styles.menuLink}
                  style={{ cursor: "not-allowed", opacity: 0.5 }}
                >
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
              <button className={styles.logoutButton} onClick={handleLogout}>
                Déconnexion
              </button>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};

export default HeaderDashboard;
