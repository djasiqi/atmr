import React, { useState } from "react";
import { Link } from "react-router-dom";
import toast from "react-hot-toast";
import styles from "./Header.module.css";

const SIGNUP_DISABLED =
  typeof process.env.REACT_APP_SIGNUP_DISABLED === "string"
    ? process.env.REACT_APP_SIGNUP_DISABLED === "true" ||
      process.env.REACT_APP_SIGNUP_DISABLED === "1"
    : true;
const COMING_SOON_TOAST_ID = "global-coming-soon";

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleComingSoon = (event) => {
    event.preventDefault();
    toast.dismiss(COMING_SOON_TOAST_ID);
    toast.custom(
      (t) => (
        <div
          className={styles.toastCard}
          role="status"
          aria-live="polite"
          data-visible={t.visible}
        >
          <div className={styles.toastIcon} aria-hidden="true">
            🚧
          </div>
          <div className={styles.toastBody}>
            <strong className={styles.toastTitle}>
              Bientôt disponible
            </strong>
            <p className={styles.toastMessage}>
              Notre équipe finalise cette section. Écrivez-nous à{" "}
              <a href="mailto:info@lirie.ch" className={styles.toastLink}>
                info@lirie.ch
              </a>{" "}
              pour être informé du lancement.
            </p>
          </div>
          <button
            type="button"
            className={styles.toastClose}
            onClick={() => toast.dismiss(COMING_SOON_TOAST_ID)}
            aria-label="Fermer"
          >
            ×
          </button>
        </div>
      ),
      {
        id: COMING_SOON_TOAST_ID,
        duration: 5000,
        position: "top-right",
        style: {
          padding: 0,
          background: "transparent",
          boxShadow: "none",
        },
      }
    );
  };

  return (
    <header className={styles.header}>
      {/* Logo */}
      <Link to="/" className={styles.logo}>
        MonTransport
      </Link>

      {/* Navigation menu */}
      <nav className={styles.nav}>
        <ul
          className={`${styles.navList} ${
            isMenuOpen ? styles.navListOpen : styles.navListClosed
          }`}
        >
          <li>
            <Link to="#" className={styles.navLink} onClick={handleComingSoon}>
              Déplacez-vous
            </Link>
          </li>
          <li>
            <Link to="#" className={styles.navLink} onClick={handleComingSoon}>
              Conduire
            </Link>
          </li>
          <li>
            <Link to="#" className={styles.navLink} onClick={handleComingSoon}>
              Professionnel
            </Link>
          </li>
          <li>
            <Link to="#" className={styles.navLink} onClick={handleComingSoon}>
              À propos
            </Link>
          </li>
          <li>
            <Link to="#" className={styles.navLink} onClick={handleComingSoon}>
              Aide
            </Link>
          </li>
        </ul>
      </nav>

      {/* Auth actions + Hamburger */}
      <div className={styles.authAndMenu}>
        <Link to="/login" className={styles.login}>
          Connexion
        </Link>
        {SIGNUP_DISABLED ? (
          <button
            type="button"
            className={`${styles.signUp} ${styles.signUpDisabled}`}
            disabled
            aria-disabled="true"
            title="Inscriptions suspendues – contactez info@lirie.ch"
          >
            S'inscrire
          </button>
        ) : (
          <button
            type="button"
            className={styles.signUp}
            onClick={handleComingSoon}
          >
            S'inscrire
          </button>
        )}
        <button
          className={styles.hamburgerButton}
          onClick={toggleMenu}
          aria-label="Toggle Navigation"
        >
          ☰
        </button>
      </div>
    </header>
  );
};

export default Header;
