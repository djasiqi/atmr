import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Header.module.css';
import { showComingSoonToast } from '../../../utils/showComingSoonToast';

const SIGNUP_DISABLED =
  typeof process.env.REACT_APP_SIGNUP_DISABLED === 'string'
    ? process.env.REACT_APP_SIGNUP_DISABLED === 'true' ||
      process.env.REACT_APP_SIGNUP_DISABLED === '1'
    : true;

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleSignUpClick = (event) => {
    event.preventDefault();
    showComingSoonToast();
  };

  return (
    <header className={styles.header}>
      {/* Logo */}
      <Link to="/" className={styles.logo}>
        <img src="/logo-lirie.png" alt="Lirie" className={styles.logoImg} width="120" height="30" />
      </Link>

      {/* Navigation menu */}
      <nav className={styles.nav}>
        <ul
          className={`${styles.navList} ${isMenuOpen ? styles.navListOpen : styles.navListClosed}`}
        >
          <li>
            <Link to="/deplacez-vous" className={styles.navLink}>
              Déplacez-vous
            </Link>
          </li>
          <li>
            <Link to="/conduire" className={styles.navLink}>
              Conduire
            </Link>
          </li>
          <li>
            <Link to="/professionnel" className={styles.navLink}>
              Professionnel
            </Link>
          </li>
          <li>
            <Link to="/a-propos" className={styles.navLink}>
              À propos
            </Link>
          </li>
          <li>
            <Link to="/aide" className={styles.navLink}>
              Aide
            </Link>
          </li>
          <li>
            <Link to="/contact" className={styles.navLink}>
              Contact
            </Link>
          </li>
        </ul>
      </nav>

      {/* Auth actions + Hamburger */}
      <div className={styles.authAndMenu}>
        <Link to="/login" className={styles.login}>
          Connexion
        </Link>
        <button
          type="button"
          className={`${styles.signUp} ${SIGNUP_DISABLED ? styles.signUpDisabled : ''}`}
          onClick={handleSignUpClick}
          aria-disabled={SIGNUP_DISABLED ? 'true' : undefined}
          title={SIGNUP_DISABLED ? 'Inscriptions suspendues – contactez info@lirie.ch' : undefined}
        >
          S'inscrire
        </button>
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
